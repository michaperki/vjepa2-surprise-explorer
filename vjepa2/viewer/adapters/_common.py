"""Shared helpers for experiment->viewer manifest adapters.

Pure, model-free utilities (git commit lookup, windowing, frame/PNG writing,
distances, the per-token surprise payload). Kept in one place so adapters don't
drift. Currently the dense `intphys_rescore` adapter is the only consumer.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from surprise_engine import CONTEXT_TOKENS_T, FRAMES_PER_CLIP, GRID_H, GRID_T, GRID_W, TUBELET_SIZE

N_TARGET_SLOTS = GRID_T - CONTEXT_TOKENS_T  # surprise time-slots over the predicted future


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def window_starts(total_frames: int, max_windows: int, stride: int) -> list[int]:
    """Evenly spaced 16-frame window starts such that each window fits in the clip."""
    span = (FRAMES_PER_CLIP - 1) * stride  # offset of the last sampled frame
    last_start = total_frames - 1 - span
    if last_start < 0:
        raise SystemExit(
            f"Video has {total_frames} frames; need at least {span + 1} for one window at stride {stride}."
        )
    if last_start == 0 or max_windows == 1:
        return [0]
    count = min(max_windows, last_start + 1)
    return sorted({int(round(s)) for s in np.linspace(0, last_start, count)})


def save_frames(frames: np.ndarray, out_dir: Path, prefix: str, run_dir: Path) -> list[str]:
    """Write frames as PNGs; return paths relative to run_dir (the manifest contract)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_paths = []
    for i, frame in enumerate(frames):
        name = f"{prefix}_f{i:02d}.png"
        iio.imwrite(out_dir / name, frame)
        rel_paths.append(str((out_dir / name).relative_to(run_dir)))
    return rel_paths


def pca_2d(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 2D PCA on `vectors`; return (mean, components, projected)."""
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    return mean, components, centered @ components.T


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def surprise_payload(token_surprise: np.ndarray, vmin: float, vmax: float) -> dict:
    """Reshape flat per-token surprise into [slot][row][col] for the viewer overlay.

    Each slot spans TUBELET_SIZE target frames; vmin/vmax are a *shared* scale so
    two heatmaps (e.g. possible vs impossible) are directly comparable.
    """
    grid = np.asarray(token_surprise, dtype=np.float32).reshape(N_TARGET_SLOTS, GRID_H, GRID_W)
    return {
        "grid": [[[round(float(v), 5) for v in row] for row in slot] for slot in grid],
        "vmin": round(vmin, 6),
        "vmax": round(vmax, 6),
        "tubelet": TUBELET_SIZE,
    }
