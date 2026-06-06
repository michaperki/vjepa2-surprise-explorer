#!/usr/bin/env python3
"""Context-length sweep — the reconciliation experiment (GPU; human launches, SOUL rule 8).

The published V-JEPA intuitive-physics result reaches 0.98 on IntPhys
(Garrido et al. 2025, arXiv:2502.11831). This repo's surprise probe sits at
chance. RECONCILIATION.md ranks the candidate causes; the top lever that is NOT
"use a bigger model" is the **context length**:

  - Their headline numbers come from a *per-property-optimal* context size, and
    they note small contexts (C=2 -> predict 14 frames) work well across
    properties.
  - This codebase only ever ran a fixed 8/8 split (CONTEXT_FRAMES=8). The
    context/target split has never been varied. The aggregation sweep varied
    pooling; the fps sweep varied temporal stride; neither touched this axis.

So this sweeps context_frames in {2,4,6,8,10,12,14} on the same IntPhys pairs,
using the same loaders/crop as the violation scorer so inputs match the model
exactly. For each context length it reports VoE accuracy under both max- and
mean-over-window aggregation, against the model-free motion baseline. If some
context length clears the motion baseline where 8/8 did not, that is the lever;
if every context length stays at chance, the null result survives the one
protocol knob the paper says matters most — a much stronger negative result.

    # smoke test (CPU, no checkpoint, exercises loaders/windowing/crop):
    PYTHONPATH=. python3 run_context_sweep.py --limit 8 --contexts 2,4,8 --dry-run
    # real (subset is plenty to see the trend; full --all is ~all 180 pairs):
    PYTHONPATH=. python3 run_context_sweep.py --limit 60 --contexts 2,4,6,8,10,12,14 --weights-dtype bf16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from run_violation_scorer import (
    all_complete_specs, crop_to_model, load_clip, motion_map, pair_movie_map,
)
from surprise_engine import FRAMES_PER_CLIP, TUBELET_SIZE


def even_window_starts(n_sampled: int, n_windows: int) -> list[int]:
    last = n_sampled - FRAMES_PER_CLIP
    if last < 0:
        return []
    if n_windows <= 1:
        return [last // 2]
    return sorted({int(round(i * last / (n_windows - 1))) for i in range(n_windows)})


def dense_window_starts(n_sampled: int, stride: int) -> list[int]:
    """All sliding-window starts at the given stride — matches the paper's dense
    per-window surprise (it averages over every stride-2 window of the movie)."""
    last = n_sampled - FRAMES_PER_CLIP
    if last < 0:
        return []
    return list(range(0, last + 1, max(1, stride)))


def movie_windows(frames: np.ndarray, n_windows: int, dense: bool, stride: int) -> list[np.ndarray]:
    starts = dense_window_starts(len(frames), stride) if dense else even_window_starts(len(frames), n_windows)
    return [frames[s:s + FRAMES_PER_CLIP] for s in starts]


def window_surprise(window: np.ndarray, context_frames: int, engine, dry_run: bool, rng) -> float:
    if dry_run:
        # Synthetic but context-dependent: shorter context => harder prediction
        # => slightly higher fake surprise, so the harness/plumbing is exercised.
        return 0.55 + 0.002 * (8 - context_frames) / 8 + 0.001 * rng.standard_normal()
    scalar, _ = engine.compute_surprise_split(window, context_frames)
    return float(scalar)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--contexts", default="2,4,6,8,10,12,14",
                    help="comma list of context frame counts (each a multiple of 2, <16)")
    ap.add_argument("--windows", type=int, default=5, help="windows per movie (ignored if --dense)")
    ap.add_argument("--dense", action="store_true",
                    help="use ALL stride-2 sliding windows (faithful to the paper) instead of N evenly-spaced")
    ap.add_argument("--stride", type=int, default=2, help="window stride when --dense")
    ap.add_argument("--frame-step", type=int, default=2,
                    help="temporal stride when sampling frames (2 ~= the paper's 7.5fps)")
    ap.add_argument("--csv", default="outputs/intphys_probe_full.csv")
    ap.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    ap.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--out", default="CONTEXT_SWEEP.md")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    contexts = [int(c) for c in args.contexts.split(",") if c.strip()]
    bad = [c for c in contexts if c % TUBELET_SIZE or not (0 < c < FRAMES_PER_CLIP)]
    if bad:
        raise SystemExit(f"invalid context sizes {bad}: each must be a multiple of {TUBELET_SIZE} in (0,{FRAMES_PER_CLIP})")

    movies = pair_movie_map(Path(args.csv))
    motion = motion_map(Path(args.csv))
    if args.pairs:
        specs = [s.strip() for s in args.pairs.split(",") if s.strip()]
    else:
        specs = all_complete_specs(movies)
    if not args.all and args.limit is not None:
        specs = specs[: args.limit]

    engine = None
    if not args.dry_run:
        import torch
        from surprise_engine import SurpriseEngine
        wd = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.weights_dtype]
        engine = SurpriseEngine(checkpoint_path=args.checkpoint, weights_dtype=wd)
        print(f"Device {engine.device}  Precision {engine.precision}")

    rng = np.random.default_rng(args.seed)

    # Model-free motion baseline on this subset (the bar any context must clear).
    mot = []
    for spec in specs:
        set_id, pid = spec.rsplit("_p", 1)
        mp = motion.get((set_id, pid, "possible")); mi = motion.get((set_id, pid, "impossible"))
        if mp is not None and mi is not None:
            mot.append(mi - mp)
    motion_acc = float((np.array(mot) > 0).mean()) if mot else float("nan")

    # Cache the cropped windows per movie once; reuse across every context length.
    win_cache: dict[str, list[np.ndarray]] = {}

    def windows_for(block, scene, movie_id):
        key = f"{block}/{scene}/{movie_id}"
        if key not in win_cache:
            frames = crop_to_model(load_clip(block, scene, movie_id, args.frame_step))
            win_cache[key] = movie_windows(frames, args.windows, args.dense, args.stride)
        return win_cache[key]

    table = []
    for ctx in contexts:
        gaps_max, gaps_mean, levels = [], [], []
        per_block: dict[str, list[bool]] = {}
        for spec in specs:
            set_id, pid = spec.rsplit("_p", 1)
            block, scene = set_id.split(":")
            mm = movies[(set_id, pid)]
            agg = {}
            for label in ("possible", "impossible"):
                vals = [window_surprise(w, ctx, engine, args.dry_run, rng)
                        for w in windows_for(block, scene, mm[label])]
                if not vals:
                    vals = [float("nan")]
                agg[label] = (max(vals), float(np.mean(vals)))
                levels.extend(vals)
            gaps_max.append(agg["impossible"][0] - agg["possible"][0])
            gaps_mean.append(agg["impossible"][1] - agg["possible"][1])
            per_block.setdefault(block, []).append(agg["impossible"][0] > agg["possible"][0])
        gaps_max = np.array(gaps_max); gaps_mean = np.array(gaps_mean)
        table.append({
            "context": ctx,
            "n": len(gaps_max),
            "mean_surprise": float(np.nanmean(levels)),
            "acc_max": float((gaps_max > 0).mean()),
            "acc_mean": float((gaps_mean > 0).mean()),
            "by_block": {b: float(np.mean(v)) for b, v in sorted(per_block.items())},
        })
        print(f"[ctx={ctx:>2}] level={table[-1]['mean_surprise']:.4f}  "
              f"acc(max)={table[-1]['acc_max']:.4f}  acc(mean)={table[-1]['acc_mean']:.4f}")

    blocks = sorted({b for t in table for b in t["by_block"]})
    win_desc = f"dense stride-{args.stride} sliding windows" if args.dense else f"{args.windows} windows/movie"
    L = ["# Context-length sweep — reconciliation with the published 0.98\n",
         f"{len(specs)} pairs, {win_desc}, frame_step={args.frame_step}. "
         f"Model-free **motion baseline = {motion_acc:.4f}** on this subset — the bar to beat.\n",
         "Per-video surprise aggregated two ways: `max` over windows (paper's preferred "
         "single-clip score) and `mean` (its pairwise score). VoE accuracy = fraction where "
         "impossible > possible.\n",
         "| context frames | predict frames | mean surprise | acc (max-agg) | acc (mean-agg) | beats motion? |",
         "| ---: | ---: | ---: | ---: | ---: | :---: |"]
    for t in table:
        best = max(t["acc_max"], t["acc_mean"])
        L.append(f"| {t['context']} | {FRAMES_PER_CLIP - t['context']} | {t['mean_surprise']:.4f} | "
                 f"{t['acc_max']:.4f} | {t['acc_mean']:.4f} | {'yes' if best > motion_acc else 'no'} |")
    if blocks:
        L.append("\n## Per-block accuracy (max-agg)\n")
        L.append("| context | " + " | ".join(blocks) + " |")
        L.append("| ---: | " + " | ".join("---:" for _ in blocks) + " |")
        for t in table:
            L.append(f"| {t['context']} | " + " | ".join(
                f"{t['by_block'].get(b, float('nan')):.3f}" for b in blocks) + " |")

    any_beats = any(max(t["acc_max"], t["acc_mean"]) > motion_acc for t in table)
    L.append("")
    if any_beats:
        winners = [t["context"] for t in table if max(t["acc_max"], t["acc_mean"]) > motion_acc]
        L.append(f"- **Some context length beats the motion baseline** (contexts {winners}). The fixed "
                 "8/8 split was hiding signal; re-run the null controls at the winning context before "
                 "claiming physics — it must also clear the equivalent-pair noise floor.")
    else:
        L.append("- **No context length clears the motion baseline.** Varying the one protocol knob the "
                 "paper says drives its IntPhys result does not recover a physics signal at ViT-L. The "
                 "remaining published-vs-ours gap is then the model itself (V-JEPA 2 ViT-L @256 here vs "
                 "V-JEPA 1 ViT-H @224 there) — an open question on this hardware, not a closed one "
                 "(SOUL rule 9), not an excuse that softens the negative result.")
    if args.dry_run:
        L = ["SYNTHETIC DRY RUN — surprise is fake; loaders/windowing/crop/context-split are real.\n"] + L
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
