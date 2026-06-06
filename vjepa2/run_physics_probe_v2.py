import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_physics_probe import (
    CONTEXT_FRAMES,
    TARGET_FRAMES,
    Appearance,
    draw_ball,
    draw_background,
    generate_pair,
    make_appearance,
    render_clip,
)
from surprise_engine import CONTEXT_TOKENS_T, GRID_H, GRID_T, GRID_W, IMG_SIZE, PATCH_SIZE, TUBELET_SIZE, SurpriseEngine


VIOLATION_TYPES = ("gravity", "continuity", "solidity")
VARIANTS = ("possible", "impossible", "linear")


def matched_gravity_triple(rng):
    app = make_appearance(rng)
    x0 = rng.uniform(62, 126)
    y0 = rng.uniform(54, 86)
    vx = rng.uniform(3.6, 6.0)
    vy = rng.uniform(2.2, 4.2)
    g = rng.uniform(0.72, 1.15)

    positions = {"possible": [], "impossible": [], "linear": []}
    for t in range(CONTEXT_FRAMES + TARGET_FRAMES):
        linear_x = x0 + vx * t
        linear_y = y0 + vy * t
        if t < CONTEXT_FRAMES:
            pos = (linear_x, linear_y)
            for variant in VARIANTS:
                positions[variant].append(pos)
            continue

        dt = t - (CONTEXT_FRAMES - 1)
        base_x = x0 + vx * (CONTEXT_FRAMES - 1)
        base_y = y0 + vy * (CONTEXT_FRAMES - 1)
        x = base_x + vx * dt
        positions["linear"].append((x, base_y + vy * dt))
        positions["possible"].append((x, base_y + vy * dt + 0.5 * g * dt * dt))
        positions["impossible"].append((x, base_y + vy * dt - 0.5 * g * dt * dt))

    return {name: render_clip(pos, app) for name, pos in positions.items()}


def continuity_triple(rng):
    possible, impossible = generate_pair("continuity", int(rng.integers(0, 2**31 - 1)))
    return {
        "possible": possible,
        "impossible": impossible,
        "linear": possible.copy(),
    }


def solidity_triple(rng):
    possible, impossible = generate_pair("solidity", int(rng.integers(0, 2**31 - 1)))
    return {
        "possible": possible,
        "impossible": impossible,
        "linear": impossible.copy(),
    }


def generate_triple(kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "gravity":
        return matched_gravity_triple(rng)
    if kind == "continuity":
        return continuity_triple(rng)
    if kind == "solidity":
        return solidity_triple(rng)
    raise ValueError(f"unknown violation type: {kind}")


def split_clip(clip):
    return clip[:CONTEXT_FRAMES], clip[CONTEXT_FRAMES:]


def assert_shared_context(clips):
    reference = clips["possible"][:CONTEXT_FRAMES]
    for name, clip in clips.items():
        if not np.array_equal(reference, clip[:CONTEXT_FRAMES]):
            raise RuntimeError(f"{name} context frames differ from possible context")


def active_token_mask(targets):
    stack = np.stack([targets[name] for name in VARIANTS], axis=0)
    diff = np.any(stack != stack[0:1], axis=(0, 4))
    active = np.zeros((GRID_T - CONTEXT_TOKENS_T, GRID_H, GRID_W), dtype=bool)
    for token_t in range(active.shape[0]):
        frame_start = token_t * TUBELET_SIZE
        frame_end = frame_start + TUBELET_SIZE
        for gy in range(GRID_H):
            y0 = gy * PATCH_SIZE
            y1 = y0 + PATCH_SIZE
            for gx in range(GRID_W):
                x0 = gx * PATCH_SIZE
                x1 = x0 + PATCH_SIZE
                active[token_t, gy, gx] = bool(diff[frame_start:frame_end, y0:y1, x0:x1].any())
    if not active.any():
        active[:, :, :] = True
    return active.reshape(-1)


def score_triple(engine, clips):
    assert_shared_context(clips)
    context, _ = split_clip(clips["possible"])
    targets = {name: split_clip(clips[name])[1] for name in VARIANTS}
    mask = active_token_mask(targets)
    outputs = engine.compute_surprises(context, [targets[name] for name in VARIANTS])

    scores = {}
    maps = {}
    for name, (all_score, token_map) in zip(VARIANTS, outputs):
        scores[name] = {
            "all": all_score,
            "local": float(np.mean(token_map[mask])),
        }
        maps[name] = token_map
    return scores, maps, mask, targets


def sem(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def save_grouped_bar(kind, result, out_path):
    means = [np.mean(result[name]["local"]) for name in VARIANTS]
    errors = [sem(result[name]["local"]) for name in VARIANTS]
    colors = ["#2b6cb0", "#c53030", "#718096"]
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    bars = ax.bar(VARIANTS, means, yerr=errors, capsize=5, color=colors)
    ax.set_ylabel("localized mean surprise")
    ax.set_title(f"{kind.capitalize()} localized active-region surprise")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_heatmap_with_mask(kind, token_map, active_mask, out_path):
    grid = token_map.reshape(GRID_T - CONTEXT_TOKENS_T, GRID_H, GRID_W)
    mask_grid = active_mask.reshape(GRID_T - CONTEXT_TOKENS_T, GRID_H, GRID_W)
    fig, axes = plt.subplots(1, grid.shape[0], figsize=(13.5, 3.6))
    vmax = float(grid.max())
    vmin = float(grid.min())
    for i, ax in enumerate(axes):
        im = ax.imshow(grid[i], cmap="magma", vmin=vmin, vmax=vmax)
        if mask_grid[i].any():
            ax.contour(mask_grid[i].astype(float), levels=[0.5], colors=["#00e5ff"], linewidths=1.6)
        ax.set_title(f"target slot {CONTEXT_TOKENS_T + i}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.91, wspace=0.08)
    cax = fig.add_axes([0.93, 0.2, 0.016, 0.62])
    fig.colorbar(im, cax=cax, label="MAE surprise")
    fig.suptitle(f"{kind.capitalize()} impossible example: surprise with active-mask outline", fontsize=13)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_example_targets(kind, targets, out_path):
    fig, axes = plt.subplots(3, TARGET_FRAMES, figsize=(15.5, 5.8))
    for row, name in enumerate(VARIANTS):
        for col in range(TARGET_FRAMES):
            ax = axes[row, col]
            ax.imshow(targets[name][col])
            ax.set_title(f"{name}\nt{col + CONTEXT_FRAMES}", fontsize=8)
            ax.axis("off")
    fig.suptitle(f"{kind.capitalize()} target continuations", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_dilution_chart(kind, result, out_path):
    local = [np.mean(result[name]["local"]) for name in VARIANTS]
    all_tokens = [np.mean(result[name]["all"]) for name in VARIANTS]
    x = np.arange(len(VARIANTS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width / 2, local, width, label="active region", color="#2b6cb0")
    ax.bar(x + width / 2, all_tokens, width, label="all target tokens", color="#a0aec0")
    ax.set_xticks(x, VARIANTS)
    ax.set_ylabel("mean surprise")
    ax.set_title(f"{kind.capitalize()} localized vs all-token surprise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_type(kind, result):
    possible = np.asarray(result["possible"]["local"])
    impossible = np.asarray(result["impossible"]["local"])
    linear = np.asarray(result["linear"]["local"])
    possible_all = np.asarray(result["possible"]["all"])
    impossible_all = np.asarray(result["impossible"]["all"])
    linear_all = np.asarray(result["linear"]["all"])
    return {
        "possible": float(possible.mean()),
        "impossible": float(impossible.mean()),
        "linear": float(linear.mean()),
        "possible_all": float(possible_all.mean()),
        "impossible_all": float(impossible_all.mean()),
        "linear_all": float(linear_all.mean()),
        "impossible_gt_possible": float(np.mean(impossible > possible)),
        "possible_lt_linear": float(np.mean(possible < linear)),
        "gap_impossible_possible": float((impossible - possible).mean()),
        "gap_linear_possible": float((linear - possible).mean()),
    }


def write_report(args, engine, results, summaries, figures):
    lines = [
        "# Controlled physics surprise probe v2",
        "",
        "Command:",
        "",
        "```bash",
        f"MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=. python3 run_physics_probe_v2.py --seeds {args.seeds}",
        "```",
        "",
        f"Device: `{engine.device}`",
        f"Precision: `{engine.precision}`",
        "",
        "Design: each trial renders possible, impossible, and naive-linear target continuations from the same 8-frame context. All three variants share byte-identical context frames and are scored with `SurpriseEngine.compute_surprises(...)`, so the V-JEPA prediction from the shared context is reused. Surprise math is unchanged: target-encoder embeddings are layer-normalized and compared to predictor outputs with mean absolute embedding error, matching V-JEPA pretraining with `loss_exp = 1.0`.",
        "",
        "Localized metric: for every trial, the active-region token mask is computed from pixel differences among the three target continuations. A target token is active if either frame in its tubelet and its 16x16 spatial patch contains any differing pixel. The same active mask is used for possible, impossible, and linear scores.",
        "",
        "Matched gravity rationale: the context moves at near-constant velocity and does not reveal acceleration direction. At the target divergence, possible and impossible continuations both receive equal-magnitude acceleration; possible accelerates downward and impossible accelerates upward. This keeps extrapolation difficulty matched while flipping physical plausibility.",
        "",
        "Caveat: these synthetic clips are out-of-distribution for V-JEPA 2. Continuity and solidity still carry some abruptness/motion-distribution confound; the naive-linear baseline is included to expose whether lower surprise reflects physical expectation or simply straight-line extrapolation. Absolute values should not be compared to natural-video runs.",
        "",
        "## Localized headline results",
        "",
        "| Type | Possible | Impossible | Linear | Impossible > possible | Possible < linear | Impossible-possible gap | Linear-possible gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for kind in results:
        s = summaries[kind]
        lines.append(
            f"| {kind} | {s['possible']:.6f} | {s['impossible']:.6f} | {s['linear']:.6f} | "
            f"{s['impossible_gt_possible']:.2%} | {s['possible_lt_linear']:.2%} | "
            f"{s['gap_impossible_possible']:.6f} | {s['gap_linear_possible']:.6f} |"
        )

    lines += [
        "",
        "## All-token dilution check",
        "",
        "| Type | Possible all-token | Impossible all-token | Linear all-token |",
        "| --- | ---: | ---: | ---: |",
    ]
    for kind in results:
        s = summaries[kind]
        lines.append(
            f"| {kind} | {s['possible_all']:.6f} | {s['impossible_all']:.6f} | {s['linear_all']:.6f} |"
        )

    lines += ["", "## Figures", ""]
    for label, path in figures:
        lines.append(f"- `{path}`: {label}")
    Path(args.report).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--report", default="PHYSICS_PROBE_V2.md")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-base", type=int, default=22000)
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
        result = {variant: {"local": [], "all": []} for variant in VARIANTS}
        example = None
        for seed_i in range(args.seeds):
            seed = args.seed_base + type_i * 1000 + seed_i
            clips = generate_triple(kind, seed)
            scores, maps, mask, targets = score_triple(engine, clips)
            for variant in VARIANTS:
                result[variant]["local"].append(scores[variant]["local"])
                result[variant]["all"].append(scores[variant]["all"])
            if example is None:
                example = (maps["impossible"], mask, targets)
            print(
                f"{kind:10s} seed={seed} "
                f"local possible={scores['possible']['local']:.6f} "
                f"impossible={scores['impossible']['local']:.6f} "
                f"linear={scores['linear']['local']:.6f} | "
                f"all possible={scores['possible']['all']:.6f} "
                f"impossible={scores['impossible']['all']:.6f} "
                f"linear={scores['linear']['all']:.6f}"
            )

        results[kind] = result
        summary = summarize_type(kind, result)
        print(
            f"{kind:10s} summary local: possible={summary['possible']:.6f} "
            f"impossible={summary['impossible']:.6f} linear={summary['linear']:.6f} "
            f"impossible>possible={summary['impossible_gt_possible']:.2%} "
            f"possible<linear={summary['possible_lt_linear']:.2%}"
        )
        print(
            f"{kind:10s} summary all: possible={summary['possible_all']:.6f} "
            f"impossible={summary['impossible_all']:.6f} linear={summary['linear_all']:.6f}"
        )

        bar_path = figures_dir / f"physics_v2_{kind}_localized_bars.png"
        heatmap_path = figures_dir / f"physics_v2_{kind}_impossible_heatmap_masked.png"
        frames_path = figures_dir / f"physics_v2_{kind}_target_examples.png"
        dilution_path = figures_dir / f"physics_v2_{kind}_localized_vs_all.png"
        save_grouped_bar(kind, result, bar_path)
        save_heatmap_with_mask(kind, example[0], example[1], heatmap_path)
        save_example_targets(kind, example[2], frames_path)
        save_dilution_chart(kind, result, dilution_path)
        figures.append((f"{kind} localized grouped bar chart for possible/impossible/linear", str(bar_path)))
        figures.append((f"{kind} impossible heatmap with active-region mask outline", str(heatmap_path)))
        figures.append((f"{kind} target-frame examples for possible/impossible/linear", str(frames_path)))
        figures.append((f"{kind} localized-vs-all-token dilution chart", str(dilution_path)))

    summaries = {kind: summarize_type(kind, result) for kind, result in results.items()}
    write_report(args, engine, results, summaries, figures)
    print(f"Wrote: {args.report}")
    print(f"Saved figures under: {figures_dir}")


if __name__ == "__main__":
    main()
