"""Violation-window review manifest for the viewer (GPU; human launches).

This is the visualization companion to `run_violation_scorer.py`. That script
scored all 180 pairs and found the result is a motion/appearance confound (the
localized surprise cancels within each IntPhys quadruplet — see
VIOLATION_SCORER.md). By default, this adapter processes every complete
possible/impossible pair in the CSV and builds a manifest rich enough to SEE
that under the hood:

  * model-cropped 256x256 MP4s of both clips (so overlays align with tokens),
  * the dense surprise curve for the playhead,
  * the located violation window (a band on the curve),
  * the active-token mask (which patches actually differ), and
  * the per-token surprise heatmap for both clips at the violation window.

The viewer then overlays the mask and the surprise heatmap on the playing video,
toggleable, so you can watch whether the model's surprise concentrates on the
violation (it doesn't — it tracks appearance). The full-population summary
(accuracy vs the motion baseline, the within-scene mirror plot) is embedded as
`manifest.analysis` from `outputs/violation_scorer.json`, so the Population tab
shows the real full-population story alongside the reviewed clips.

    # CPU end-to-end test (synthetic surprise, real localizer/mask/MP4):
    PYTHONPATH=. python3 -m viewer.adapters.violation_review --dry-run --out runs/violation_review
    # real full-population run (the human launches once the GPU is free):
    PYTHONPATH=. python3 -m viewer.adapters.violation_review --out runs/violation_review --weights-dtype bf16
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_violation_scorer import (
    SHORT_SIDE, crop_to_model, divergence_curve, pair_movie_map,
)
from surprise_engine import (
    CONTEXT_FRAMES, FRAMES_PER_CLIP, GRID_H, GRID_W, IMG_SIZE, PATCH_SIZE, TUBELET_SIZE,
)
from viewer.adapters._common import git_commit
from viewer.adapters.intphys_rescore import all_complete_specs, encode_mp4, window_starts
from viewer.inventory import write_inventory
from viewer.manifest import Manifest

TARGET_SLOTS = (FRAMES_PER_CLIP - CONTEXT_FRAMES) // TUBELET_SIZE  # 4
WINDOW_TARGET_CENTER = CONTEXT_FRAMES + (FRAMES_PER_CLIP - CONTEXT_FRAMES - 1) / 2  # 11.5
# The model resizes a frame to SHORT_SIDE then center-crops IMG_SIZE, so its
# token grid covers only the center (IMG_SIZE/SHORT_SIDE) of the frame we encode.
# The frontend insets the overlay by this fraction so patches line up with tokens.
OVERLAY_INSET = round((1 - IMG_SIZE / SHORT_SIDE) / 2, 4)

def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:d}:{sec:02d}"


def scene_specs(scenes: list[str], movies: dict) -> list[str]:
    out = []
    for (set_id, pair_id), mm in sorted(movies.items()):
        if set_id in scenes and "possible" in mm and "impossible" in mm:
            out.append(f"{set_id}_p{pair_id}")
    return out


def synth_grid(mask16: np.ndarray, seed: int, impossible: bool) -> np.ndarray:
    """Fake 16x16 per-window surprise map (CPU dry-run): baseline + a bump in the
    patches that differ, for the impossible clip only — exercises the overlay."""
    rng = np.random.default_rng(seed)
    g = 0.56 + rng.normal(0, 0.008, (GRID_H, GRID_W))
    if impossible:
        g[mask16] += 0.06
    return g


def window_mask(pos_tgt: np.ndarray, imp_tgt: np.ndarray, pixel_thresh: float = 10.0) -> np.ndarray:
    """16x16 bool: patches that genuinely differ between the two clips anywhere in
    this window's 8 target frames (per-pixel change above the ~0.3 rendering-noise
    floor). No all-true fallback — an empty mask honestly means 'identical here'."""
    diff = np.abs(pos_tgt.astype(np.float32) - imp_tgt.astype(np.float32)).mean(axis=-1)  # (8,256,256)
    changed = diff > pixel_thresh
    m = np.zeros((GRID_H, GRID_W), dtype=bool)
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            m[gy, gx] = changed[:, gy * PATCH_SIZE:(gy + 1) * PATCH_SIZE, gx * PATCH_SIZE:(gx + 1) * PATCH_SIZE].any()
    return m


def score_timeline(engine, frames, starts, masks, impossible, dry_run, seed):
    """Per window: (scalar surprise, 16x16 surprise map averaged over the window's
    target tubelets). These are the SAME forward passes the dense curve already
    needs — keeping the grid is free — so the heatmap can ride the playhead across
    the whole clip instead of one pre-chosen window."""
    scalars, grids = [], []
    for k, s in enumerate(starts):
        if dry_run:
            g = synth_grid(masks[k], (seed + k) & 0xFFFFFFFF, impossible)
        else:
            win = frames[s:s + FRAMES_PER_CLIP]
            _, tok = engine.compute_surprise(win[:CONTEXT_FRAMES], win[CONTEXT_FRAMES:])
            g = np.asarray(tok, dtype=float).reshape(TARGET_SLOTS, GRID_H, GRID_W).mean(axis=0)
        scalars.append(round(float(g.mean()), 6))
        grids.append(g)
    return scalars, grids


def build_example(spec, movies, engine, out_dir: Path, fps: int, stride: int, dry_run: bool) -> dict:
    set_id, pid = spec.rsplit("_p", 1)
    block, scene = set_id.split(":")
    mm = movies[(set_id, pid)]
    safe = spec.replace(":", "-")
    assets = out_dir / "assets" / safe
    assets.mkdir(parents=True, exist_ok=True)

    from run_violation_scorer import load_clip
    pos = crop_to_model(load_clip(block, scene, mm["possible"], 1))
    imp = crop_to_model(load_clip(block, scene, mm["impossible"], 1))
    n = min(len(pos), len(imp))
    pos, imp = pos[:n], imp[:n]

    # stride>1 subsamples the windows: fewer forward passes and a lighter manifest,
    # still dense enough for a smooth playhead-following heatmap.
    starts = window_starts(n, stride)
    centers = [round(s + WINDOW_TARGET_CENTER, 1) for s in starts]
    masks = [window_mask(pos[s + CONTEXT_FRAMES:s + FRAMES_PER_CLIP],
                         imp[s + CONTEXT_FRAMES:s + FRAMES_PER_CLIP]) for s in starts]
    pos_curve, pos_grids = score_timeline(engine, pos, starts, masks, False, dry_run, hash((spec, "p")) & 0xFFFF)
    imp_curve, imp_grids = score_timeline(engine, imp, starts, masks, True, dry_run, hash((spec, "i")) & 0xFFFF)

    # Honest pixel-difference signal (where the two clips differ), per window —
    # NOT a claim about where the physical violation is.
    pdiff_frame = divergence_curve(pos, imp)
    pdiff = [round(float(pdiff_frame[s:s + FRAMES_PER_CLIP].mean()), 5) for s in starts]
    peak_frame = int(np.argmax(pdiff_frame))

    vmin = float(min(min(g.min() for g in pos_grids), min(g.min() for g in imp_grids)))
    vmax = float(max(max(g.max() for g in pos_grids), max(g.max() for g in imp_grids)))
    pos_mean, imp_mean = float(np.mean(pos_curve)), float(np.mean(imp_curve))

    encode_mp4(pos, assets / "possible.mp4", fps, IMG_SIZE)
    encode_mp4(imp, assets / "impossible.mp4", fps, IMG_SIZE)

    def g2(grid):  # 16x16 -> rounded nested list (3 dp keeps ~100 color levels, compact)
        return [[round(float(v), 3) for v in row] for row in grid]

    return {
        "id": spec,
        "label": "correct" if imp_mean > pos_mean else "wrong",
        "video_possible": str((assets / "possible.mp4").relative_to(out_dir)),
        "video_impossible": str((assets / "impossible.mp4").relative_to(out_dir)),
        "fps": fps, "n_frames": int(n),
        "metrics": {
            "surprise_gap": round(imp_mean - pos_mean, 6),
            "possible_mean": round(pos_mean, 6),
            "impossible_mean": round(imp_mean, 6),
            "max_pixeldiff_frame": peak_frame,
            "n_windows": len(starts),
        },
        "dense_curve": {"center": centers, "possible": pos_curve, "impossible": imp_curve},
        "pixel_diff": {"center": centers, "value": pdiff},
        "heatmap": {
            "center": centers,
            "possible": [g2(g) for g in pos_grids],
            "impossible": [g2(g) for g in imp_grids],
            "mask": [[[int(b) for b in row] for row in m] for m in masks],
            "vmin": round(vmin, 4), "vmax": round(vmax, 4),
            "grid_h": GRID_H, "grid_w": GRID_W, "overlay_inset": OVERLAY_INSET,
        },
    }


def build_analysis(records_path: Path) -> dict | None:
    if not records_path.exists():
        return None
    recs = json.loads(records_path.read_text())
    gap = np.array([r["gap_localized"] for r in recs])
    mot = np.array([r["motion_diff"] for r in recs])
    by_scene: dict[str, list[float]] = defaultdict(list)
    for r in recs:
        by_scene[r.get("scene", r["spec"].rsplit("_p", 1)[0])].append(r["gap_localized"])
    mirror = [[round(v[0], 5), round(v[1], 5)] for v in by_scene.values() if len(v) == 2]
    opp = float(np.mean([(a > 0) != (b > 0) for a, b in mirror])) if mirror else float("nan")
    return {
        "source": str(records_path),
        "n_pairs": len(recs),
        "accuracy_localized": round(float((gap > 0).mean()), 4),
        "accuracy_motion": round(float((mot > 0).mean()), 4),
        "anti_symmetry_pct": round(opp, 4),
        "mirror": mirror,
        "note": "Full 180-pair violation-window result. accuracy_localized is the model; "
                "accuracy_motion is the model-free motion baseline it must beat. Each mirror "
                "point is a scene's two pair-gaps; they pile on y=-x where they cancel.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", help="optional subset: comma list like O1:01_p2,O1:15_p4")
    ap.add_argument("--scenes", help="optional subset: comma list of scenes; both pairs of each are included")
    ap.add_argument("--csv", default="outputs/intphys_probe_full.csv")
    ap.add_argument("--records", default="outputs/violation_scorer.json",
                    help="full-population scorer output to embed as manifest.analysis")
    ap.add_argument("--out", default="runs/violation_review")
    ap.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    ap.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--stride", type=int, default=2,
                    help="window hop for the dense curve + heatmap timeline (2 = every other window)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    movies = pair_movie_map(Path(args.csv))
    if args.pairs:
        specs = [s.strip() for s in args.pairs.split(",") if s.strip()]
        selection = "explicit pair subset"
    elif args.scenes:
        specs = scene_specs([s.strip() for s in args.scenes.split(",")], movies)
        selection = "scene subset"
    else:
        specs = all_complete_specs(movies)
        selection = "all complete pairs"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[review] selected {len(specs)} pair(s) ({selection}); out={out_dir} stride={args.stride}",
          flush=True)

    engine = None
    if not args.dry_run:
        import torch
        from surprise_engine import SurpriseEngine
        wd = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.weights_dtype]
        engine = SurpriseEngine(checkpoint_path=args.checkpoint, weights_dtype=wd)
        print(f"Device {engine.device}  Precision {engine.precision}")

    examples = []
    started_at = time.monotonic()
    for idx, spec in enumerate(specs, start=1):
        pair_started_at = time.monotonic()
        print(f"[review {idx}/{len(specs)}] {spec} ... elapsed {format_duration(pair_started_at - started_at)}",
              flush=True)
        ex = build_example(spec, movies, engine, out_dir, args.fps, args.stride, args.dry_run)
        pair_elapsed = time.monotonic() - pair_started_at
        mt = ex["metrics"]
        print(f"  {mt['n_windows']} windows  gap(whole-clip)={mt['surprise_gap']:+.5f}  "
              f"max-pixel-diff@{mt['max_pixeldiff_frame']}  pair={format_duration(pair_elapsed)}  "
              f"total={format_duration(time.monotonic() - started_at)}",
              flush=True)
        examples.append(ex)

    notes = (f"Violation-window review of {len(examples)} {selection} (model-cropped 256 MP4s). The surprise "
             "heatmap rides the playhead across the WHOLE clip (per-window 16x16 maps — same "
             "forward passes as the dense curve, so no extra compute); the pixel-difference curve "
             "shows where the two clips differ, which is NOT necessarily the physical violation. "
             "Full-180 result embedded as analysis.")
    if args.dry_run:
        notes = "SYNTHETIC DRY RUN — surprise + heatmaps are fake (MP4/localizer/mask real). " + notes
    m = Manifest(run_id=out_dir.name, notes=notes, commit=git_commit(),
                 config_path=None if args.dry_run else args.checkpoint,
                 command=f"python3 -m viewer.adapters.violation_review --out {args.out}")
    m.examples = examples
    m.analysis = build_analysis(Path(args.records))
    path = m.write(out_dir)
    inventory_json, inventory_csv = write_inventory(out_dir, Path(args.csv), Path(args.records))
    print(f"Wrote {path} with {len(examples)} pair(s)"
          + (f"; analysis from {args.records}" if m.analysis else "; no analysis (records missing)"))
    print(f"Wrote {inventory_json} and {inventory_csv}")


if __name__ == "__main__":
    main()
