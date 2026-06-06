#!/usr/bin/env python3
"""Violation-window surprise scorer (GPU; the human launches it per SOUL rule 8).

The whole-clip metrics fail because the matched possible/impossible movies are
~pixel-identical except for a brief, spatially-small violation, so a surprise
averaged (or maxed) over all windows and all tokens is dominated by identical
content. NULL_CONTROLS.md + AGGREGATION_SWEEP.md show no *pooling* of the
existing surprise beats a motion baseline. The one thing left to try is a
*different surprise*: score only the tokens, in only the time window, where the
two clips actually differ.

The localizer is **independent of the model**: it comes from the raw pixel
divergence between the frame-aligned possible and impossible clips. That is not
circular with surprise — we use pixels to choose *where* to look, then ask the
model how surprised it is *there*.

Per pair:
  1. Crop both clips to the model's 256 input (resize short side 292 + center crop).
  2. Per-frame divergence between possible and impossible -> locate the violation
     frame; place a 16-frame window so the violation lands in the target half.
  3. Spatial token mask = the 16x16-per-tubelet patches that differ in the target
     half. Score surprise for each clip, average over the masked target tokens.
  4. gap = surprise_impossible - surprise_possible, localized in space and time.

Evaluated against the bars from the null controls: chance (0.5) and the
**motion-only baseline (0.583)**. Finer localization only matters if it clears them.

    # smoke test (4 pairs) once the GPU is free:
    PYTHONPATH=. python3 run_violation_scorer.py --limit 4 --weights-dtype bf16
    # full population:
    PYTHONPATH=. python3 run_violation_scorer.py --all --weights-dtype bf16
    # CPU geometry test (no model): synthetic surprise, real localizer/window/mask:
    PYTHONPATH=. python3 run_violation_scorer.py --limit 4 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from surprise_engine import (
    CONTEXT_FRAMES, CONTEXT_TOKENS_T, FRAMES_PER_CLIP, GRID_H, GRID_T, GRID_W,
    IMG_SIZE, PATCH_SIZE, TUBELET_SIZE,
)

SHORT_SIDE = int(256.0 / 224 * IMG_SIZE)  # 292; matches build_video_transform
TARGET_SLOTS = GRID_T - CONTEXT_TOKENS_T  # 4
WINDOW_TARGET_CENTER = CONTEXT_FRAMES + (FRAMES_PER_CLIP - CONTEXT_FRAMES - 1) / 2  # 11.5


def pair_movie_map(csv_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for r in csv.DictReader(csv_path.open()):
        out[(r["set_id"], r["pair_id"])][r["label"]] = r["movie_id"]
    return out


def all_complete_specs(movies: dict) -> list[str]:
    return sorted(f"{s}_p{p}" for (s, p), mm in movies.items()
                  if "possible" in mm and "impossible" in mm)


def motion_map(csv_path: Path) -> dict[tuple[str, str, str], float]:
    return {(r["set_id"], r["pair_id"], r["label"]): float(r["motion_energy"])
            for r in csv.DictReader(csv_path.open())}


def load_clip(block: str, scene: str, movie_id: str, frame_step: int) -> np.ndarray:
    src = Path("dev") / block / scene / str(movie_id) / "scene"
    pngs = sorted(p for p in src.glob("scene_*.png"))
    if not pngs:
        raise SystemExit(f"No frames at {src}")
    frames = np.stack([np.asarray(Image.open(p).convert("RGB")) for p in pngs], axis=0)
    return frames[::frame_step]


def crop_to_model(frames: np.ndarray) -> np.ndarray:
    """(T,H,W,3) uint8 -> (T,256,256,3): resize short side to 292, center crop 256,
    matching build_video_transform so patch indices line up with model tokens."""
    out = []
    for fr in frames:
        img = Image.fromarray(fr)
        w, h = img.size
        if w <= h:
            nw, nh = SHORT_SIDE, round(h * SHORT_SIDE / w)
        else:
            nw, nh = round(w * SHORT_SIDE / h), SHORT_SIDE
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        left = (nw - IMG_SIZE) // 2
        top = (nh - IMG_SIZE) // 2
        out.append(np.asarray(img.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))))
    return np.stack(out, axis=0)


def divergence_curve(pos: np.ndarray, imp: np.ndarray) -> np.ndarray:
    """Per-frame mean abs pixel difference between the aligned clips (cropped)."""
    n = min(len(pos), len(imp))
    return np.abs(pos[:n].astype(np.float32) - imp[:n].astype(np.float32)).mean(axis=(1, 2, 3))


def locate_violation(curve: np.ndarray) -> int:
    """Peak divergence frame of the violation, independent of the model.

    The violation is a localized pixel-divergence event; we want the *peak*
    frame (so it lands in the scored target half), but robust to stray noise
    spikes. We find the largest contiguous run above (median + 3*MAD) and return
    the frame of maximum divergence within it; if nothing clears the threshold,
    the global argmax."""
    if curve.max() <= 0:
        return len(curve) // 2
    med = float(np.median(curve))
    mad = float(np.median(np.abs(curve - med))) or 1e-6
    above = curve > med + 3.0 * mad
    if not above.any():
        return int(curve.argmax())
    best_lo, best_hi, best_len, lo = 0, 1, -1, None
    for i, a in enumerate(list(above) + [False]):
        if a and lo is None:
            lo = i
        elif not a and lo is not None:
            if (i - lo) > best_len:
                best_lo, best_hi, best_len = lo, i, i - lo
            lo = None
    return int(best_lo + curve[best_lo:best_hi].argmax())


def window_start(center: int, n_frames: int) -> int:
    return int(np.clip(round(center - WINDOW_TARGET_CENTER), 0, max(0, n_frames - FRAMES_PER_CLIP)))


def spatial_token_mask(pos_tgt: np.ndarray, imp_tgt: np.ndarray, pixel_thresh: float = 10.0) -> np.ndarray:
    """Flat (TARGET_SLOTS*GRID_H*GRID_W,) bool mask of target tokens whose 16x16
    patch genuinely differs between the clips. The rendered possible/impossible
    movies differ by sub-pixel noise *everywhere*, so an exact `!=` marks every
    patch active; we instead require a per-pixel intensity change above
    `pixel_thresh` (default 10/255) — well above the ~0.3 rendering-noise floor —
    so the mask localizes to the violation region, not the whole frame."""
    # mean abs diff across RGB per pixel, per target frame -> (8,256,256)
    diff = np.abs(pos_tgt.astype(np.float32) - imp_tgt.astype(np.float32)).mean(axis=-1)
    changed = diff > pixel_thresh
    active = np.zeros((TARGET_SLOTS, GRID_H, GRID_W), dtype=bool)
    for t in range(TARGET_SLOTS):
        f0, f1 = t * TUBELET_SIZE, t * TUBELET_SIZE + TUBELET_SIZE
        for gy in range(GRID_H):
            for gx in range(GRID_W):
                patch = changed[f0:f1, gy * PATCH_SIZE:(gy + 1) * PATCH_SIZE, gx * PATCH_SIZE:(gx + 1) * PATCH_SIZE]
                active[t, gy, gx] = bool(patch.any())
    if not active.any():
        active[:] = True  # no patch cleared threshold -> fall back to all tokens
    return active.reshape(-1)


def score_pair(spec, movies, engine, motion, frame_step, dry_run, rng) -> dict:
    set_id, pid = spec.rsplit("_p", 1)
    block, scene = set_id.split(":")
    mm = movies[(set_id, pid)]
    pos = crop_to_model(load_clip(block, scene, mm["possible"], frame_step))
    imp = crop_to_model(load_clip(block, scene, mm["impossible"], frame_step))
    curve = divergence_curve(pos, imp)
    center = locate_violation(curve)
    start = window_start(center, min(len(pos), len(imp)))
    sl = slice(start, start + FRAMES_PER_CLIP)
    pos_win, imp_win = pos[sl], imp[sl]
    mask = spatial_token_mask(pos_win[CONTEXT_FRAMES:], imp_win[CONTEXT_FRAMES:])

    def surprise(win):
        if dry_run:
            tok = rng.normal(0.56, 0.02, TARGET_SLOTS * GRID_H * GRID_W)
            return float(tok.mean()), tok
        return engine.compute_surprise(win[:CONTEXT_FRAMES], win[CONTEXT_FRAMES:])

    pos_scalar, pos_tok = surprise(pos_win)
    imp_scalar, imp_tok = surprise(imp_win)
    pos_loc = float(np.asarray(pos_tok)[mask].mean())
    imp_loc = float(np.asarray(imp_tok)[mask].mean())
    m_pos = motion.get((set_id, pid, "possible"), float("nan"))
    m_imp = motion.get((set_id, pid, "impossible"), float("nan"))
    return {
        "spec": spec, "scene": set_id, "block": block,
        "violation_center": center, "window_start": start, "n_active_tokens": int(mask.sum()),
        "pos_loc": pos_loc, "imp_loc": imp_loc,
        "gap_localized": imp_loc - pos_loc,
        "gap_alltoken": imp_scalar - pos_scalar,
        "motion_diff": m_imp - m_pos,
    }


def summarize(records: list[dict]) -> str:
    from scipy import stats
    g_loc = np.array([r["gap_localized"] for r in records])
    g_all = np.array([r["gap_alltoken"] for r in records])
    mot = np.array([r["motion_diff"] for r in records])
    n = len(records)

    def line(name, g):
        k = int((g > 0).sum())
        p = float(stats.binomtest(k, n, 0.5).pvalue) if n else float("nan")
        return f"| {name} | {k/n:.4f} | {k}/{n} | {p:.4f} | {np.median(np.abs(g)):.5f} |"

    mot_acc = float((mot > 0).mean())
    L = ["# Violation-window scorer results\n",
         f"{n} pairs. Surprise scored only in the pixel-divergence-localized "
         "space-time window. Bars: chance 0.5, motion-only baseline "
         f"{mot_acc:.4f}.\n",
         "| metric | accuracy | k/n | p vs 0.5 | median |gap| |",
         "| --- | ---: | ---: | ---: | ---: |",
         line("violation-localized", g_loc),
         line("all-token (violation window)", g_all),
         f"| motion-only baseline | {mot_acc:.4f} | — | — | — |", ""]
    loc_acc = float((g_loc > 0).mean())
    verdict = ("beats" if loc_acc > mot_acc else "does NOT beat")
    L.append(f"**Verdict:** violation-localized accuracy {loc_acc:.4f} {verdict} the motion "
             f"baseline {mot_acc:.4f}. "
             + ("A real, model-contributed physics signal — worth pursuing." if loc_acc > mot_acc
                else "Finer localization does not rescue the result; the null result holds even "
                     "when scoring exactly where the violation is."))

    # Within-scene anti-symmetry: IntPhys scenes have two matched pairs. If the
    # localized surprise tracks appearance (which clip shows the object in the
    # masked patch) rather than physical possibility, the two pairs' gaps cancel
    # (gap_a ≈ -gap_b) and accuracy is pinned near 0.5 by construction, no matter
    # how well the violation is localized. This checks for exactly that.
    by_scene: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_scene[r.get("scene", r["spec"].rsplit("_p", 1)[0])].append(r["gap_localized"])
    two = [v for v in by_scene.values() if len(v) == 2]
    if two:
        sums = np.array([a + b for a, b in two])
        gaps2 = np.array([abs(x) for v in two for x in v])
        opp = float(np.mean([(a > 0) != (b > 0) for a, b in two]))
        L.append("")
        L.append("**Within-scene anti-symmetry check** "
                 f"({len(two)} scenes with two pairs): the two gaps have opposite sign in "
                 f"{opp*100:.0f}% of scenes; median |gap_a + gap_b| = {np.median(np.abs(sums)):.5f} "
                 f"vs median |gap| = {np.median(gaps2):.5f} "
                 f"(ratio {float(np.median(np.abs(sums)))/max(1e-9, float(np.median(gaps2))):.2f}). "
                 + ("Near-perfect cancellation: the localized surprise is measuring an "
                    "appearance difference that is symmetric across the matched quadruplet, not "
                    "physical possibility — so it cannot exceed chance by construction. This is "
                    "the motion/appearance confound resurfacing under tight localization."
                    if (opp > 0.8 and np.median(np.abs(sums)) < 0.5 * np.median(gaps2))
                    else "Gaps do not systematically cancel within scenes, so the metric is not "
                         "trivially appearance-symmetric; the accuracy above is the real test."))
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", help="Comma list like O1:15_p4,O2:08_p1")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--csv", default="outputs/intphys_probe_full.csv")
    ap.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    ap.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    ap.add_argument("--frame-step", type=int, default=1)
    ap.add_argument("--out", default="VIOLATION_SCORER.md")
    ap.add_argument("--records", default="outputs/violation_scorer.json")
    ap.add_argument("--dry-run", action="store_true", help="synthetic surprise; real localizer (CPU)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    movies = pair_movie_map(Path(args.csv))
    motion = motion_map(Path(args.csv))
    if args.pairs:
        specs = [s.strip() for s in args.pairs.split(",") if s.strip()]
    else:
        specs = all_complete_specs(movies)  # --all is the default; --limit caps it
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
    records = []
    for spec in specs:
        rec = score_pair(spec, movies, engine, motion, args.frame_step, args.dry_run, rng)
        print(f"[score] {spec:12s} center={rec['violation_center']:3d} "
              f"start={rec['window_start']:3d} active={rec['n_active_tokens']:4d} "
              f"gap_loc={rec['gap_localized']:+.5f}")
        records.append(rec)

    Path(args.records).parent.mkdir(parents=True, exist_ok=True)
    Path(args.records).write_text(json.dumps(records, indent=2) + "\n")
    report = summarize(records)
    if args.dry_run:
        report = "SYNTHETIC DRY RUN — surprise is fake; localizer/window/mask are real.\n\n" + report
    Path(args.out).write_text(report + "\n")
    print("\n" + report)
    print(f"\nWrote {args.out} and {args.records}")


if __name__ == "__main__":
    main()
