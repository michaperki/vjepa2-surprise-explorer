import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from run_intphys_probe import (
    CONTEXT_FRAMES,
    FRAMES_PER_CLIP,
    active_mask_for_targets,
    discover_intphys2_debug,
    discover_original_intphys,
    label_from_type,
    read_original_intphys_variant_windows,
    read_video_mp4_windows,
    variant_pair_id,
)
from surprise_engine import CONTEXT_TOKENS_T, GRID_H, GRID_T, GRID_W


def read_csv_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def parse_scores(value):
    return np.asarray([float(x) for x in value.split()], dtype=np.float64)


def row_key(row):
    return (row["set_id"], row["pair_id"], row["label"])


def pair_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["set_id"], row["pair_id"])][row["label"]] = row
    pairs = []
    for (set_id, pair_id), pair in grouped.items():
        if "possible" not in pair or "impossible" not in pair:
            continue
        possible = pair["possible"]
        impossible = pair["impossible"]
        gap = float(impossible["aggregated_surprise_localized"]) - float(possible["aggregated_surprise_localized"])
        pairs.append(
            {
                "set_id": set_id,
                "pair_id": pair_id,
                "possible": possible,
                "impossible": impossible,
                "gap": gap,
                "abs_gap": abs(gap),
                "correct": gap > 0,
                "differentiable": possible.get("differentiable") == "True",
            }
        )
    return pairs


def select_pairs(pairs, limit):
    selected = []
    categories = [
        ("tie", sorted(pairs, key=lambda p: p["abs_gap"])),
        ("wrong", sorted([p for p in pairs if p["gap"] < 0], key=lambda p: p["gap"])),
        ("correct", sorted([p for p in pairs if p["gap"] > 0], key=lambda p: -p["gap"])),
    ]
    seen = set()
    for category, candidates in categories:
        for pair in candidates:
            key = (pair["set_id"], pair["pair_id"])
            if key in seen:
                continue
            selected.append((category, pair))
            seen.add(key)
            break
        if len(selected) >= limit:
            return selected
    for pair in sorted(pairs, key=lambda p: p["abs_gap"]):
        key = (pair["set_id"], pair["pair_id"])
        if key not in seen:
            selected.append(("extra", pair))
            seen.add(key)
        if len(selected) >= limit:
            break
    return selected


def build_groups(args):
    if args.source == "intphys2-debug":
        groups = discover_intphys2_debug(args.data_root, max_sets=args.max_sets)
        read_windows = lambda row: read_video_mp4_windows(
            row["local_path"],
            frame_step=args.frame_step,
            stride=args.stride,
            max_windows=args.max_windows,
            decode_short_side=args.decode_short_side,
        )
    else:
        groups = discover_original_intphys(
            args.data_root,
            blocks=args.blocks,
            max_sets=args.max_sets,
            max_sets_per_block=args.max_sets_per_block,
        )
        read_windows = lambda row: read_original_intphys_variant_windows(
            row["local_path"],
            frame_step=args.frame_step,
            stride=args.stride,
            max_windows=args.max_windows,
            decode_short_side=args.decode_short_side,
        )
    return groups, read_windows


def find_dataset_row(groups, csv_row):
    rows = groups[csv_row["set_id"]]
    for row in rows:
        label = label_from_type(row["type"])
        movie_id = row.get("name") or Path(row["local_path"]).stem
        if label == csv_row["label"] and movie_id == csv_row["movie_id"]:
            return row
    for row in rows:
        label = label_from_type(row["type"])
        pair_id = variant_pair_id(row["type"])
        if label == csv_row["label"] and pair_id == csv_row["pair_id"]:
            return row
    raise KeyError(f"could not find dataset row for {csv_row['set_id']} {csv_row['pair_id']} {csv_row['label']}")


def target_diff_image(pos_window, imp_window):
    pos_target = pos_window[CONTEXT_FRAMES:]
    imp_target = imp_window[CONTEXT_FRAMES:]
    diff = np.mean(np.abs(pos_target.astype(np.float32) - imp_target.astype(np.float32)), axis=(0, 3))
    if diff.max() > 0:
        diff = diff / diff.max()
    return diff


def active_grid(pos_window, imp_window):
    pos_target = pos_window[CONTEXT_FRAMES:]
    imp_target = imp_window[CONTEXT_FRAMES:]
    active = active_mask_for_targets(pos_target, imp_target)
    if active.size == 0:
        return np.zeros((GRID_T - CONTEXT_TOKENS_T, GRID_H, GRID_W), dtype=bool)
    return active.reshape(GRID_T - CONTEXT_TOKENS_T, GRID_H, GRID_W)


def save_pair_dashboard(category, pair, groups, read_windows, out_path):
    pos_row = pair["possible"]
    imp_row = pair["impossible"]
    pos_data = find_dataset_row(groups, pos_row)
    imp_data = find_dataset_row(groups, imp_row)
    pos_windows, _, _, pos_footprint = read_windows(pos_data)
    imp_windows, _, _, imp_footprint = read_windows(imp_data)

    pos_scores = parse_scores(pos_row["window_surprises_localized"])
    imp_scores = parse_scores(imp_row["window_surprises_localized"])
    gaps = imp_scores - pos_scores
    if len(gaps) == 0:
        window_idx = 0
    elif category == "tie":
        window_idx = int(np.argmin(np.abs(gaps)))
    elif category == "wrong":
        window_idx = int(np.argmin(gaps))
    else:
        window_idx = int(np.argmax(gaps))
    window_idx = min(window_idx, len(pos_windows) - 1, len(imp_windows) - 1)

    pos_window = pos_windows[window_idx]
    imp_window = imp_windows[window_idx]
    diff = target_diff_image(pos_window, imp_window)
    mask = active_grid(pos_window, imp_window)

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(4, 8, height_ratios=[1.0, 1.0, 1.1, 1.1])

    frame_indices = np.linspace(0, FRAMES_PER_CLIP - 1, 8).round().astype(int)
    for col, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(pos_window[frame_idx])
        ax.set_title(f"possible f{frame_idx}", fontsize=8)
        ax.axis("off")
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(imp_window[frame_idx])
        ax.set_title(f"impossible f{frame_idx}", fontsize=8)
        ax.axis("off")

    ax_trace = fig.add_subplot(grid[2, :4])
    x = np.arange(len(pos_scores))
    ax_trace.plot(x, pos_scores, marker="o", label="possible", color="#2b6cb0")
    ax_trace.plot(x, imp_scores, marker="o", label="impossible", color="#c53030")
    ax_trace.axvline(window_idx, color="#111827", linestyle="--", linewidth=1)
    ax_trace.set_title("Localized per-window surprise trace")
    ax_trace.set_xlabel("window")
    ax_trace.set_ylabel("surprise")
    ax_trace.legend()

    ax_gap = fig.add_subplot(grid[2, 4:])
    ax_gap.bar(x, gaps, color=np.where(gaps > 0, "#2f855a", "#c53030"))
    ax_gap.axhline(0, color="#111827", linewidth=1)
    ax_gap.axvline(window_idx, color="#111827", linestyle="--", linewidth=1)
    ax_gap.set_title("Impossible - possible gap per window")
    ax_gap.set_xlabel("window")
    ax_gap.set_ylabel("gap")

    ax_diff = fig.add_subplot(grid[3, :4])
    im = ax_diff.imshow(diff, cmap="magma")
    ax_diff.set_title("Target pixel difference averaged over target frames")
    ax_diff.axis("off")
    fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.02)

    ax_mask = fig.add_subplot(grid[3, 4:])
    mask_sum = mask.sum(axis=0)
    im2 = ax_mask.imshow(mask_sum, cmap="viridis", vmin=0, vmax=max(1, mask.shape[0]))
    ax_mask.set_title(f"Active-token mask coverage ({int(mask.sum())} / {mask.size} target tokens)")
    ax_mask.axis("off")
    fig.colorbar(im2, ax=ax_mask, fraction=0.046, pad=0.02)

    title = (
        f"{category.upper()} diagnostic: {pair['set_id']} pair {pair['pair_id']} | "
        f"agg gap={pair['gap']:.6f} | differentiable={pair['differentiable']} | "
        f"read pos {pos_footprint['frames_read']}/{pos_footprint['num_frames']}, "
        f"imp {imp_footprint['frames_read']}/{imp_footprint['num_frames']}"
    )
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["intphys2-debug", "intphys-dev"], default="intphys-dev")
    parser.add_argument("--data-root", default="dev")
    parser.add_argument("--blocks", nargs="+", default=["O1", "O2", "O3"])
    parser.add_argument("--max-sets", type=int, default=30)
    parser.add_argument("--max-sets-per-block", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--decode-short-side", type=int, default=320)
    parser.add_argument("--csv", default="outputs/intphys_probe_movies.csv")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    rows = read_csv_rows(args.csv)
    pairs = pair_rows(rows)
    selected = select_pairs(pairs, args.limit)
    groups, read_windows = build_groups(args)
    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, (category, pair) in enumerate(selected, start=1):
        out_path = out_dir / f"intphys_diagnostic_{idx}_{category}_{pair['set_id'].replace(':', '_')}_p{pair['pair_id']}.png"
        save_pair_dashboard(category, pair, groups, read_windows, out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
