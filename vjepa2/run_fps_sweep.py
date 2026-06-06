#!/usr/bin/env python3
"""Frame-rate / distribution-shift sweep (GPU; the human launches it, SOUL rule 8).

Surprise sits near ~0.56 for essentially every clip, possible or impossible
(NULL_CONTROLS.md). One explanation is distribution shift: IntPhys clips sampled
at frame_step=1 may move much slower than the clips V-JEPA was pretrained on, so
the predictor is uniformly uncertain and has no headroom to be *more* surprised
by a violation. This sweep tests that directly: re-score the same pairs at
several temporal strides and watch two things —

  (a) the absolute surprise level. If it's flat across strides, the model is
      saturated/OOD everywhere and frame rate isn't the lever. If it dips at some
      stride, that's the closer-to-training sampling.
  (b) VoE accuracy vs the motion-only baseline at each stride. Matching the frame
      rate only matters if accuracy clears the baseline somewhere it didn't before.

To bound GPU cost, each movie is scored at a fixed number of evenly-spaced
16-frame windows (default 5) and averaged. Reuses the loaders/crop from the
violation scorer so the inputs match the model exactly.

    # smoke test:
    PYTHONPATH=. python3 run_fps_sweep.py --limit 12 --steps 1,2,3,4 --dry-run
    # real (subset is plenty to see the trend):
    PYTHONPATH=. python3 run_fps_sweep.py --limit 60 --steps 1,2,3,4,6 --weights-dtype bf16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_violation_scorer import (
    all_complete_specs, crop_to_model, load_clip, motion_map, pair_movie_map,
)
from surprise_engine import CONTEXT_FRAMES, FRAMES_PER_CLIP


def even_window_starts(n_sampled: int, n_windows: int) -> list[int]:
    last = n_sampled - FRAMES_PER_CLIP
    if last < 0:
        return []
    if n_windows <= 1:
        return [last // 2]
    return sorted({int(round(i * last / (n_windows - 1))) for i in range(n_windows)})


def movie_surprise(frames: np.ndarray, n_windows: int, engine, dry_run: bool, rng) -> float:
    starts = even_window_starts(len(frames), n_windows)
    if not starts:
        return float("nan")
    vals = []
    for s in starts:
        win = frames[s:s + FRAMES_PER_CLIP]
        if dry_run:
            # synthetic level rises slightly with effective motion (fewer frames
            # spanned per window => faster) so the sweep harness is exercised.
            vals.append(0.55 + 0.001 * rng.standard_normal())
        else:
            scalar, _ = engine.compute_surprise(win[:CONTEXT_FRAMES], win[CONTEXT_FRAMES:])
            vals.append(float(scalar))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--steps", default="1,2,3,4,6", help="comma list of frame_step values")
    ap.add_argument("--windows", type=int, default=5, help="windows per movie to average")
    ap.add_argument("--csv", default="outputs/intphys_probe_full.csv")
    ap.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    ap.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--out", default="FPS_SWEEP.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    movies = pair_movie_map(Path(args.csv))
    motion = motion_map(Path(args.csv))
    if args.all:
        specs = all_complete_specs(movies)
    elif args.pairs:
        specs = [s.strip() for s in args.pairs.split(",") if s.strip()]
    else:
        specs = all_complete_specs(movies)
    if args.limit is not None:
        specs = specs[: args.limit]

    engine = None
    if not args.dry_run:
        import torch
        from surprise_engine import SurpriseEngine
        wd = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.weights_dtype]
        engine = SurpriseEngine(checkpoint_path=args.checkpoint, weights_dtype=wd)
        print(f"Device {engine.device}  Precision {engine.precision}")

    rng = np.random.default_rng(args.seed)
    # motion baseline on this subset
    mot = []
    for spec in specs:
        set_id, pid = spec.rsplit("_p", 1)
        mp = motion.get((set_id, pid, "possible")); mi = motion.get((set_id, pid, "impossible"))
        if mp is not None and mi is not None:
            mot.append(mi - mp)
    motion_acc = float((np.array(mot) > 0).mean())

    # cache raw native frames per movie so we resample per stride without re-reading PNGs
    raw: dict[str, np.ndarray] = {}
    table = []
    for step in steps:
        gaps, levels = [], []
        for spec in specs:
            set_id, pid = spec.rsplit("_p", 1)
            block, scene = set_id.split(":")
            mm = movies[(set_id, pid)]
            s_vals = {}
            for label in ("possible", "impossible"):
                key = f"{block}/{scene}/{mm[label]}"
                if key not in raw:
                    raw[key] = load_clip(block, scene, mm[label], 1)
                frames = crop_to_model(raw[key][::step])
                s = movie_surprise(frames, args.windows, engine, args.dry_run, rng)
                s_vals[label] = s
                levels.append(s)
            gaps.append(s_vals["impossible"] - s_vals["possible"])
        gaps = np.array(gaps)
        table.append({
            "step": step, "n": len(gaps),
            "mean_surprise": float(np.nanmean(levels)),
            "accuracy": float((gaps > 0).mean()),
        })
        print(f"[sweep] step={step}  mean_surprise={table[-1]['mean_surprise']:.4f}  "
              f"acc={table[-1]['accuracy']:.4f}")

    L = ["# Frame-rate / distribution-shift sweep\n",
         f"{len(specs)} pairs, {args.windows} windows/movie averaged. "
         f"Motion-only baseline accuracy on this subset: **{motion_acc:.4f}**.\n",
         "| frame_step | mean surprise level | VoE accuracy | beats motion? |",
         "| ---: | ---: | ---: | :---: |"]
    for t in table:
        L.append(f"| {t['step']} | {t['mean_surprise']:.4f} | {t['accuracy']:.4f} | "
                 f"{'yes' if t['accuracy'] > motion_acc else 'no'} |")
    lvl = [t["mean_surprise"] for t in table]
    spread = max(lvl) - min(lvl)
    L.append("")
    L.append(f"- Surprise-level spread across strides: **{spread:.4f}**. "
             + ("Flat — the model is saturated/OOD at every frame rate; frame rate is not the "
                "lever." if spread < 0.02 else
                "The level moves with stride; the lowest-surprise stride is the closest to "
                "V-JEPA's training distribution and the fairest setting to re-evaluate."))
    if not any(t["accuracy"] > motion_acc for t in table):
        L.append("- No stride clears the motion baseline: distribution shift in frame rate does "
                 "not recover a physics signal.")
    if args.dry_run:
        L = ["SYNTHETIC DRY RUN — surprise is fake; sampling/crop/windowing are real.\n"] + L
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
