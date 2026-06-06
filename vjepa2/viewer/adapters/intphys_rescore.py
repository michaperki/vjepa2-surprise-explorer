"""Dense per-pair re-score + MP4 for the synced video/playhead player (GPU job).

The powered run (`run_intphys_probe.py`) scores each movie at only 12 sliding
windows (frame_step=2, stride=2) and stores scalars — too coarse to localize
*where* in the clip the surprise peaks. This adapter takes a curated handful of
pairs and re-scores them densely (frame_step=1, stride=1 -> ~85 windows over the
full 100-frame clip), encodes an MP4 of the exact frames the model saw, and
emits a manifest whose examples carry video + a dense surprise curve aligned to
the video timeline. The viewer then plays both clips in lockstep with a playhead
riding the curve.

Per SOUL.md principle 8 this loads the model, so the human launches it once the
main run is off the GPU. `--dry-run` fakes the surprise (real MP4 still encoded)
so the player can be tested end-to-end on CPU.

A curated handful:

    PYTHONPATH=. python3 -m viewer.adapters.intphys_rescore \
        --pairs O1:15_p4,O1:15_p1,O2:06_p3,O2:08_p1,O3:16_p4,O3:12_p4 \
        --out runs/intphys_rescore --weights-dtype bf16

Or the whole population (every complete possible/impossible pair in the CSV);
`--limit` caps it for a smoke test before committing to the full ~180-pair run:

    PYTHONPATH=. python3 -m viewer.adapters.intphys_rescore \
        --all --limit 4 --out runs/intphys_rescore_all --weights-dtype bf16   # smoke test
    PYTHONPATH=. python3 -m viewer.adapters.intphys_rescore \
        --all --out runs/intphys_rescore_all --weights-dtype bf16             # full population
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image

from viewer.adapters._common import git_commit
from viewer.manifest import Manifest

CONTEXT_FRAMES = 8
TARGET_FRAMES = 8
FRAMES_PER_CLIP = CONTEXT_FRAMES + TARGET_FRAMES  # 16
# Surprise at window w refers to its target half; this is its center, in the
# units of the played (sampled) frame sequence, so the playhead lands on it.
WINDOW_TARGET_CENTER = CONTEXT_FRAMES + (TARGET_FRAMES - 1) / 2  # 11.5


def pair_movie_map(csv_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """(set_id, pair_id) -> {'possible': movie_id, 'impossible': movie_id}."""
    out: dict[tuple[str, str], dict[str, str]] = {}
    for r in csv.DictReader(csv_path.open()):
        out.setdefault((r["set_id"], r["pair_id"]), {})[r["label"]] = r["movie_id"]
    return out


def all_complete_specs(movies: dict[tuple[str, str], dict[str, str]]) -> list[str]:
    """Every (set_id, pair_id) that has both a possible and an impossible movie,
    as `{set_id}_p{pair_id}` specs — the same format `--pairs` accepts."""
    specs = [
        f"{set_id}_p{pair_id}"
        for (set_id, pair_id), mm in movies.items()
        if "possible" in mm and "impossible" in mm
    ]
    return sorted(specs)


def load_clip(block: str, scene: str, movie_id: str) -> np.ndarray:
    """All frames of one clip as a (T, H, W, 3) uint8 array."""
    src = Path("dev") / block / scene / str(movie_id) / "scene"
    pngs = sorted(p for p in src.glob("scene_*.png"))
    if not pngs:
        raise SystemExit(f"No frames at {src}")
    return np.stack([np.asarray(Image.open(p).convert("RGB")) for p in pngs], axis=0)


def window_starts(n_sampled: int, stride: int) -> list[int]:
    if n_sampled < FRAMES_PER_CLIP:
        return []
    return list(range(0, n_sampled - FRAMES_PER_CLIP + 1, stride))


def dense_surprise(engine, sampled: np.ndarray, starts: list[int]) -> list[float]:
    vals = []
    for s in starts:
        window = sampled[s : s + FRAMES_PER_CLIP]
        scalar, _ = engine.compute_surprise(window[:CONTEXT_FRAMES], window[CONTEXT_FRAMES:])
        vals.append(round(float(scalar), 6))
    return vals


def synthetic_surprise(starts: list[int], seed: int, bump: bool) -> list[float]:
    """Deterministic fake curve for --dry-run: a baseline + (optional) mid bump."""
    rng = np.random.default_rng(seed)
    base = 0.55 + 0.004 * np.sin(np.linspace(0, 3.0, len(starts))) + rng.normal(0, 0.0015, len(starts))
    if bump and len(starts):
        c = len(starts) * 0.6
        base += 0.03 * np.exp(-0.5 * ((np.arange(len(starts)) - c) / max(1, len(starts) * 0.08)) ** 2)
    return [round(float(v), 6) for v in base]


def encode_mp4(sampled: np.ndarray, path: Path, fps: int, width: int) -> None:
    frames = []
    for fr in sampled:
        img = Image.fromarray(fr)
        w, h = img.size
        frames.append(np.asarray(img.resize((width, max(1, round(h * width / w))))))
    iio.mimwrite(path, frames, fps=fps, codec="libx264", macro_block_size=16, quality=7)


def build_pair(spec: str, movies: dict, engine, out_dir: Path, fps: int, width: int,
               frame_step: int, stride: int, dry_run: bool) -> dict:
    set_id, pid = spec.rsplit("_p", 1)
    block, scene = set_id.split(":")
    mm = movies.get((set_id, pid))
    if not mm or "possible" not in mm or "impossible" not in mm:
        raise SystemExit(f"{spec}: no possible/impossible movie mapping in CSV")

    safe = spec.replace(":", "-")
    assets = out_dir / "assets" / safe
    assets.mkdir(parents=True, exist_ok=True)

    curve = {}
    n_frames = fps_used = 0
    for label in ("possible", "impossible"):
        sampled = load_clip(block, scene, mm[label])[::frame_step]
        starts = window_starts(len(sampled), stride)
        if not starts:
            raise SystemExit(f"{spec}/{label}: clip too short for one window")
        if dry_run:
            vals = synthetic_surprise(starts, seed=hash((spec, label)) & 0xFFFF, bump=(label == "impossible"))
        else:
            vals = dense_surprise(engine, sampled, starts)
        encode_mp4(sampled, assets / f"{label}.mp4", fps, width)
        curve[label] = vals
        curve["center"] = [round(s + WINDOW_TARGET_CENTER, 1) for s in starts]
        n_frames, fps_used = len(sampled), fps

    pos_mean, imp_mean = float(np.mean(curve["possible"])), float(np.mean(curve["impossible"]))
    correct = imp_mean > pos_mean
    return {
        "id": spec,
        "label": "correct" if correct else "wrong",
        "metrics": {
            "surprise_gap": round(imp_mean - pos_mean, 6),
            "possible": round(pos_mean, 6),
            "impossible": round(imp_mean, 6),
            "n_windows": len(curve["center"]),
        },
        "video_possible": str((assets / "possible.mp4").relative_to(out_dir)),
        "video_impossible": str((assets / "impossible.mp4").relative_to(out_dir)),
        "fps": fps_used,
        "n_frames": n_frames,
        "dense_curve": {
            "center": curve["center"],
            "possible": curve["possible"],
            "impossible": curve["impossible"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", help="Comma list like O1:15_p4,O2:08_p1")
    parser.add_argument("--all", action="store_true",
                        help="Score every complete possible/impossible pair in the CSV")
    parser.add_argument("--limit", type=int, default=None,
                        help="With --all, cap the number of pairs (smoke test before the full run)")
    parser.add_argument("--csv", default="outputs/intphys_probe_full.csv", help="Powered CSV for movie mapping")
    parser.add_argument("--out", default="runs/intphys_rescore")
    parser.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    parser.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--frame-step", type=int, default=1, help="1 = every native frame (densest)")
    parser.add_argument("--stride", type=int, default=1, help="window hop over sampled frames")
    parser.add_argument("--fps", type=int, default=12, help="playback fps for the encoded MP4")
    parser.add_argument("--width", type=int, default=256, help="MP4 width (aspect preserved)")
    parser.add_argument("--dry-run", action="store_true", help="synthetic surprise, real MP4 (CPU test)")
    args = parser.parse_args()

    movies = pair_movie_map(Path(args.csv))
    if args.all:
        specs = all_complete_specs(movies)
        if args.limit is not None:
            specs = specs[: args.limit]
        print(f"[rescore] --all: {len(specs)} complete pair(s) from {args.csv}")
    elif args.pairs:
        specs = [s.strip() for s in args.pairs.split(",") if s.strip()]
    else:
        raise SystemExit("pass --pairs <list> or --all")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = None
    if not args.dry_run:
        import torch

        from surprise_engine import SurpriseEngine

        wd = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.weights_dtype]
        engine = SurpriseEngine(checkpoint_path=args.checkpoint, weights_dtype=wd)
        print(f"Device {engine.device}  Precision {engine.precision}")

    examples = []
    for spec in specs:
        print(f"[rescore] {spec} ...", flush=True)
        ex = build_pair(spec, movies, engine, out_dir, args.fps, args.width,
                        args.frame_step, args.stride, args.dry_run)
        print(f"  {ex['label']:7s} gap {ex['metrics']['surprise_gap']:+.5f}  "
              f"{ex['metrics']['n_windows']} windows  {ex['n_frames']} frames")
        examples.append(ex)

    n_correct = sum(1 for e in examples if e["label"] == "correct")
    accuracy = n_correct / len(examples) if examples else float("nan")
    print(f"Population: {n_correct}/{len(examples)} correct (impossible > possible) "
          f"= accuracy {accuracy:.4f}")

    selection = "all complete" if args.all else "curated"
    notes = (
        f"Dense per-pair re-score (frame_step={args.frame_step}, stride={args.stride}) of {len(examples)} "
        f"{selection} pairs, with MP4 of the model-seen frames and a surprise curve aligned to the video "
        f"timeline. Population accuracy (impossible > possible mean surprise): {n_correct}/{len(examples)} "
        f"= {accuracy:.4f}. Denser than the powered run (frame_step=2/stride=2/12 windows), so absolute "
        f"values differ — the per-clip curve localizes where surprise rises; the population accuracy is "
        f"the headline VoE number."
    )
    if args.dry_run:
        notes = "SYNTHETIC DRY RUN — surprise curve is fake (MP4 is real). " + notes
    manifest = Manifest(
        run_id=Path(args.out).name,
        command=f"PYTHONPATH=. python3 -m viewer.adapters.intphys_rescore --pairs {args.pairs} --out {args.out}",
        notes=notes,
        config_path=None if args.dry_run else args.checkpoint,
        commit=git_commit(),
    )
    manifest.examples = examples
    path = manifest.write(out_dir)
    print(f"Wrote {path} with {len(examples)} pair(s).")


if __name__ == "__main__":
    main()
