import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from decord import VideoReader

from surprise_engine import CONTEXT_FRAMES, CONTEXT_TOKENS_T, FRAMES_PER_CLIP, GRID_H, GRID_T, GRID_W, SurpriseEngine


def load_sampled_clip(video_path):
    vr = VideoReader(str(video_path))
    frame_idx = np.arange(FRAMES_PER_CLIP) * 4
    if frame_idx[-1] >= len(vr):
        frame_idx = np.linspace(0, len(vr) - 1, FRAMES_PER_CLIP).astype(np.int64)
    frames = vr.get_batch(frame_idx).asnumpy()
    return frames, frame_idx, len(vr)


def make_broken_target(frames):
    replacement_order = np.arange(CONTEXT_FRAMES - 1, -1, -1)
    return frames[replacement_order]


def save_contact_sheet(frames, frame_idx, output_path):
    fig, axes = plt.subplots(2, 8, figsize=(16, 4.6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(frames[i])
        role = "context" if i < CONTEXT_FRAMES else "target"
        ax.set_title(f"{role}\nframe {int(frame_idx[i])}", fontsize=9)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(3)
            spine.set_edgecolor("#2b6cb0" if role == "context" else "#c53030")
    fig.suptitle("Sampled clip frames: context first, target future second", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_heatmaps(real_map, broken_map, output_path):
    real_grid = real_map.reshape(GRID_T - CONTEXT_TOKENS_T, GRID_H, GRID_W)
    broken_grid = broken_map.reshape(GRID_T - CONTEXT_TOKENS_T, GRID_H, GRID_W)
    vmax = max(float(real_grid.max()), float(broken_grid.max()))
    vmin = min(float(real_grid.min()), float(broken_grid.min()))

    fig, axes = plt.subplots(2, real_grid.shape[0], figsize=(14, 6))
    for row, (name, grid) in enumerate((("real future", real_grid), ("broken future", broken_grid))):
        for t in range(grid.shape[0]):
            ax = axes[row, t]
            im = ax.imshow(grid[t], cmap="magma", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}\ntarget slot {CONTEXT_TOKENS_T + t}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.subplots_adjust(right=0.92, hspace=0.35, wspace=0.08)
    cax = fig.add_axes([0.94, 0.16, 0.015, 0.68])
    fig.colorbar(im, cax=cax, label="mean absolute embedding error")
    fig.suptitle("Per-token V-JEPA 2 surprise heatmaps", fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_bar_chart(real_scalar, broken_scalar, output_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["real future", "broken future"], [real_scalar, broken_scalar], color=["#2b6cb0", "#c53030"])
    ax.set_ylabel("mean token surprise")
    ax.set_title("Scalar surprise comparison")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{height:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_report(args, engine, real_scalar, broken_scalar, held, figures):
    report = f"""# Surprise probe run

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=. python3 run_surprise_probe.py
```

Device: `{engine.device}`
Precision: `{engine.precision}`

Context/target split: 16 sampled frames become 8 temporal token slots with tubelet size 2. Context uses slots 0-3, covering sampled frames 0-7. Target uses slots 4-7, covering sampled frames 8-15.

Training-distance match: V-JEPA pretraining uses `mean(abs(predicted - target) ** loss_exp) / loss_exp` with `loss_exp = 1.0`, after layer-normalizing target-encoder embeddings. This probe reports mean absolute embedding error per target token, then averages over target tokens for the scalar.

Scalar surprise:

- Real future: `{real_scalar:.6f}`
- Broken future: `{broken_scalar:.6f}`
- Expected `real < broken`: `{"held" if held else "did not hold"}`

Figures:

- `{figures["contact"]}`: contact sheet of sampled frames, labeled context vs. target.
- `{figures["heatmap"]}`: per-target-token surprise reshaped into four 16x16 target-time heatmaps for real and broken futures.
- `{figures["bars"]}`: bar chart comparing scalar surprise for real vs. broken future.
"""
    Path(args.report).write_text(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/sample_video.mp4")
    parser.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--report", default="SURPRISE_RUN.md")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames, frame_idx, total_frames = load_sampled_clip(Path(args.video))
    context = frames[:CONTEXT_FRAMES]
    real_target = frames[CONTEXT_FRAMES:]
    broken_target = make_broken_target(frames)

    engine = SurpriseEngine(checkpoint_path=args.checkpoint)
    print(f"Device: {engine.device}")
    print(f"Precision: {engine.precision}")
    print(f"Video: {args.video}")
    print(f"Frames sampled: {frame_idx.tolist()} from {total_frames} total frames")
    print(f"Context token slots: 0-{CONTEXT_TOKENS_T - 1}; target token slots: {CONTEXT_TOKENS_T}-{GRID_T - 1}")

    real_scalar, real_map = engine.compute_surprise(context, real_target)
    broken_scalar, broken_map = engine.compute_surprise(context, broken_target)
    held = real_scalar < broken_scalar

    contact_path = figures_dir / "surprise_contact_sheet.png"
    heatmap_path = figures_dir / "surprise_heatmaps.png"
    bars_path = figures_dir / "surprise_bars.png"
    save_contact_sheet(frames, frame_idx, contact_path)
    save_heatmaps(real_map, broken_map, heatmap_path)
    save_bar_chart(real_scalar, broken_scalar, bars_path)

    print(f"Real future surprise:   {real_scalar:.6f}")
    print(f"Broken future surprise: {broken_scalar:.6f}")
    print(f"Expectation real < broken: {'held' if held else 'did not hold'}")
    print(f"Saved: {contact_path}")
    print(f"Saved: {heatmap_path}")
    print(f"Saved: {bars_path}")

    write_report(
        args,
        engine,
        real_scalar,
        broken_scalar,
        held,
        {
            "contact": str(contact_path),
            "heatmap": str(heatmap_path),
            "bars": str(bars_path),
        },
    )
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
