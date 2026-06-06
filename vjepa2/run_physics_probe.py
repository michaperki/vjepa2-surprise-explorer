import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from surprise_engine import (
    CONTEXT_FRAMES,
    CONTEXT_TOKENS_T,
    GRID_H,
    GRID_T,
    GRID_W,
    IMG_SIZE,
    TARGET_FRAMES,
    SurpriseEngine,
)


VIOLATION_TYPES = ("gravity", "continuity", "solidity")


@dataclass
class Appearance:
    bg_a: np.ndarray
    bg_b: np.ndarray
    ball_color: np.ndarray
    radius: float
    texture_phase: float
    floor_y: float | None = None


def make_appearance(rng, floor_y=None):
    bg_a = np.array(
        [
            rng.uniform(32, 78),
            rng.uniform(42, 88),
            rng.uniform(70, 124),
        ],
        dtype=np.float32,
    )
    bg_b = np.array(
        [
            rng.uniform(108, 170),
            rng.uniform(112, 180),
            rng.uniform(126, 205),
        ],
        dtype=np.float32,
    )
    ball_color = np.array(
        [
            rng.uniform(185, 245),
            rng.uniform(62, 130),
            rng.uniform(46, 105),
        ],
        dtype=np.float32,
    )
    return Appearance(
        bg_a=bg_a,
        bg_b=bg_b,
        ball_color=ball_color,
        radius=float(rng.uniform(27, 36)),
        texture_phase=float(rng.uniform(0, 2 * np.pi)),
        floor_y=floor_y,
    )


def draw_background(app):
    yy = np.linspace(0, 1, IMG_SIZE, dtype=np.float32)[:, None, None]
    xx = np.linspace(0, 1, IMG_SIZE, dtype=np.float32)[None, :, None]
    bg = app.bg_a * (1 - yy) + app.bg_b * yy
    texture = 10.0 * np.sin(2 * np.pi * (xx * 3.0 + yy * 1.4) + app.texture_phase)
    bg = bg + texture
    if app.floor_y is not None:
        floor = int(app.floor_y)
        bg[floor:, :, :] *= np.array([0.55, 0.58, 0.62], dtype=np.float32)
        bg[max(0, floor - 3) : floor + 3, :, :] = np.array([215, 218, 224], dtype=np.float32)
    return np.clip(bg, 0, 255)


def draw_ball(frame, center, radius, color):
    y, x = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    cx, cy = center
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = np.clip((radius + 0.8 - dist) / 1.6, 0, 1)[..., None]
    shade = 0.58 + 0.42 * np.clip((-(x - cx) - (y - cy)) / (radius * 1.8) + 0.65, 0, 1)
    shaded = color[None, None, :] * shade[..., None]
    frame[:] = frame * (1 - alpha) + shaded * alpha
    highlight = np.exp(-(((x - (cx - radius * 0.35)) ** 2 + (y - (cy - radius * 0.42)) ** 2) / (2 * (radius * 0.16) ** 2)))
    frame[:] = np.clip(frame + highlight[..., None] * 45.0 * alpha, 0, 255)


def render_clip(positions, app):
    frames = []
    for pos in positions:
        frame = draw_background(app).copy()
        draw_ball(frame, pos, app.radius, app.ball_color)
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return np.stack(frames, axis=0)


def gravity_pair(rng):
    app = make_appearance(rng)
    x0 = rng.uniform(72, 134)
    y0 = rng.uniform(34, 58)
    vx = rng.uniform(2.0, 5.8)
    vy = rng.uniform(1.6, 3.4)
    g = rng.uniform(0.62, 1.05)

    possible, impossible = [], []
    for t in range(CONTEXT_FRAMES + TARGET_FRAMES):
        x = x0 + vx * t
        y = y0 + vy * t + 0.5 * g * t * t
        possible.append((x, y))
        if t < CONTEXT_FRAMES:
            impossible.append((x, y))
        else:
            dt = t - (CONTEXT_FRAMES - 1)
            last_x, last_y = possible[CONTEXT_FRAMES - 1]
            impossible.append((last_x + vx * dt, last_y - rng.uniform(1.1, 2.1) * dt))
    return render_clip(possible, app), render_clip(impossible, app)


def continuity_pair(rng):
    app = make_appearance(rng)
    x0 = rng.uniform(42, 78)
    y0 = rng.uniform(88, 160)
    vx = rng.uniform(6.0, 8.5)
    vy = rng.uniform(-1.2, 1.2)
    jump = np.array([rng.uniform(54, 82), rng.uniform(-48, 48)], dtype=np.float32)

    possible, impossible = [], []
    for t in range(CONTEXT_FRAMES + TARGET_FRAMES):
        pos = np.array([x0 + vx * t, y0 + vy * t], dtype=np.float32)
        possible.append(tuple(pos))
        if t < CONTEXT_FRAMES:
            impossible.append(tuple(pos))
        else:
            impossible.append(tuple(pos + jump))
    return render_clip(possible, app), render_clip(impossible, app)


def solidity_pair(rng):
    floor_y = rng.uniform(178, 196)
    app = make_appearance(rng, floor_y=floor_y)
    x0 = rng.uniform(62, 130)
    vx = rng.uniform(2.8, 5.4)
    y0 = floor_y - app.radius - rng.uniform(82, 104)
    vy = rng.uniform(7.8, 10.0)
    bounce_t = CONTEXT_FRAMES + rng.integers(1, 3)

    possible, impossible = [], []
    for t in range(CONTEXT_FRAMES + TARGET_FRAMES):
        x = x0 + vx * t
        y_linear = y0 + vy * t
        impossible.append((x, y_linear))
        if t < bounce_t:
            y = min(y_linear, floor_y - app.radius)
        else:
            y = floor_y - app.radius - (vy * 0.82) * (t - bounce_t)
        possible.append((x, y))

    # Match the context exactly, then diverge only in the target frames.
    impossible[:CONTEXT_FRAMES] = possible[:CONTEXT_FRAMES]
    return render_clip(possible, app), render_clip(impossible, app)


def generate_pair(kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "gravity":
        return gravity_pair(rng)
    if kind == "continuity":
        return continuity_pair(rng)
    if kind == "solidity":
        return solidity_pair(rng)
    raise ValueError(f"unknown violation type: {kind}")


def split_clip(clip):
    return clip[:CONTEXT_FRAMES], clip[CONTEXT_FRAMES:]


def score_pair(engine, possible_clip, impossible_clip):
    context, possible_target = split_clip(possible_clip)
    impossible_context, impossible_target = split_clip(impossible_clip)
    if not np.array_equal(context, impossible_context):
        raise RuntimeError("minimal-pair context frames are not identical")
    (possible_score, possible_map), (impossible_score, impossible_map) = engine.compute_surprises(
        context,
        [possible_target, impossible_target],
    )
    return possible_score, impossible_score, possible_map, impossible_map


def sem(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def save_type_bar(kind, possible, impossible, out_path):
    means = [float(np.mean(possible)), float(np.mean(impossible))]
    errors = [sem(possible), sem(impossible)]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    bars = ax.bar(["possible", "impossible"], means, yerr=errors, capsize=5, color=["#2b6cb0", "#c53030"])
    ax.set_ylabel("mean target-token surprise")
    ax.set_title(f"{kind.capitalize()} minimal pairs")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_heatmap(kind, token_map, out_path):
    grid = token_map.reshape(GRID_T - CONTEXT_TOKENS_T, GRID_H, GRID_W)
    fig, axes = plt.subplots(1, grid.shape[0], figsize=(13.5, 3.6))
    vmax = float(grid.max())
    vmin = float(grid.min())
    for i, ax in enumerate(axes):
        im = ax.imshow(grid[i], cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(f"target slot {CONTEXT_TOKENS_T + i}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.91, wspace=0.08)
    cax = fig.add_axes([0.93, 0.2, 0.016, 0.62])
    fig.colorbar(im, cax=cax, label="MAE surprise")
    fig.suptitle(f"{kind.capitalize()} impossible example: per-token surprise", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_example_pair(kind, possible_clip, impossible_clip, out_path):
    fig, axes = plt.subplots(2, 8, figsize=(15.5, 4.4))
    for row, (name, clip) in enumerate((("possible", possible_clip), ("impossible", impossible_clip))):
        for col in range(8):
            frame_i = CONTEXT_FRAMES + col
            ax = axes[row, col]
            ax.imshow(clip[frame_i])
            ax.set_title(f"{name}\nf{frame_i}", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{kind.capitalize()} target-window minimal pair", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_overall_summary(results, out_path):
    kinds = list(results)
    possible = [np.mean(results[k]["possible"]) for k in kinds]
    impossible = [np.mean(results[k]["impossible"]) for k in kinds]
    x = np.arange(len(kinds))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - width / 2, possible, width, label="possible", color="#2b6cb0")
    ax.bar(x + width / 2, impossible, width, label="impossible", color="#c53030")
    ax.set_xticks(x, [k.capitalize() for k in kinds])
    ax.set_ylabel("mean target-token surprise")
    ax.set_title("Overall physics-pair surprise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_report(args, engine, results, figures):
    lines = [
        "# Controlled physics surprise probe",
        "",
        "Command:",
        "",
        "```bash",
        f"MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=. python3 run_physics_probe.py --seeds {args.seeds}",
        "```",
        "",
        f"Device: `{engine.device}`",
        f"Precision: `{engine.precision}`",
        "",
        "Design: each trial renders a possible/impossible synthetic minimal pair. The first 8 frames are byte-identical and are used as context. The last 8 frames diverge into a physically possible or impossible target. The V-JEPA 2 predictor receives context token slots 0-3 and predicts target token slots 4-7; surprise is the same layer-normalized target-embedding mean absolute error used by the V-JEPA pretraining loss with `loss_exp = 1.0`.",
        "",
        "Caveat: these synthetic clips are out-of-distribution for V-JEPA 2. Only within-pair possible-vs-impossible comparisons are meaningful; absolute values should not be compared to natural-video runs.",
        "",
        "## Results",
        "",
        "| Type | Possible mean | Impossible mean | Gap | Impossible > possible |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind, data in results.items():
        possible = np.asarray(data["possible"])
        impossible = np.asarray(data["impossible"])
        gap = impossible.mean() - possible.mean()
        rate = np.mean(impossible > possible)
        lines.append(
            f"| {kind} | {possible.mean():.6f} | {impossible.mean():.6f} | {gap:.6f} | {rate:.2%} |"
        )
    lines += ["", "## Figures", ""]
    for label, path in figures:
        lines.append(f"- `{path}`: {label}")
    Path(args.report).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--report", default="PHYSICS_PROBE.md")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=12000)
    parser.add_argument("--types", nargs="+", default=list(VIOLATION_TYPES), choices=VIOLATION_TYPES)
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    engine = SurpriseEngine(checkpoint_path=args.checkpoint)
    print(f"Device: {engine.device}")
    print(f"Precision: {engine.precision}")
    print(f"Seeds per type: {args.seeds}")

    results = {}
    figures = []
    for type_i, kind in enumerate(args.types):
        possible_scores, impossible_scores = [], []
        example = None
        for seed_i in range(args.seeds):
            seed = args.seed_base + type_i * 1000 + seed_i
            possible_clip, impossible_clip = generate_pair(kind, seed)
            possible_score, impossible_score, possible_map, impossible_map = score_pair(
                engine, possible_clip, impossible_clip
            )
            possible_scores.append(possible_score)
            impossible_scores.append(impossible_score)
            if example is None:
                example = (possible_clip, impossible_clip, impossible_map)
            print(
                f"{kind:10s} seed={seed} possible={possible_score:.6f} "
                f"impossible={impossible_score:.6f} gap={impossible_score - possible_score:.6f}"
            )

        possible_arr = np.asarray(possible_scores, dtype=np.float64)
        impossible_arr = np.asarray(impossible_scores, dtype=np.float64)
        rate = float(np.mean(impossible_arr > possible_arr))
        gap = float(np.mean(impossible_arr - possible_arr))
        results[kind] = {"possible": possible_arr, "impossible": impossible_arr}
        print(
            f"{kind:10s} summary: possible_mean={possible_arr.mean():.6f} "
            f"impossible_mean={impossible_arr.mean():.6f} gap={gap:.6f} "
            f"impossible>possible={rate:.2%}"
        )

        bar_path = figures_dir / f"physics_{kind}_bars.png"
        heatmap_path = figures_dir / f"physics_{kind}_impossible_heatmap.png"
        example_path = figures_dir / f"physics_{kind}_example_pair.png"
        save_type_bar(kind, possible_arr, impossible_arr, bar_path)
        save_heatmap(kind, example[2], heatmap_path)
        save_example_pair(kind, example[0], example[1], example_path)
        figures.append((f"{kind} possible-vs-impossible mean surprise bar chart with error bars", str(bar_path)))
        figures.append((f"{kind} impossible example target-token heatmap", str(heatmap_path)))
        figures.append((f"{kind} possible/impossible target-window example frames", str(example_path)))

    overall_path = figures_dir / "physics_overall_summary.png"
    save_overall_summary(results, overall_path)
    figures.append(("overall per-type possible-vs-impossible summary", str(overall_path)))
    write_report(args, engine, results, figures)
    print(f"Wrote: {args.report}")
    print(f"Saved figures under: {figures_dir}")


if __name__ == "__main__":
    main()
