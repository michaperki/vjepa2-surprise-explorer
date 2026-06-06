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
    SurpriseEngine,
    active_mask_for_targets,
    discover_intphys2_debug,
    discover_original_intphys,
    label_from_type,
    read_original_intphys_variant_windows,
    read_video_mp4_windows,
    variant_pair_id,
)


SHUFFLE_ORDER = [3, 0, 7, 1, 6, 2, 5, 4]


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def pair_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["set_id"], row["pair_id"])][row["label"]] = row
    pairs = []
    for (set_id, pair_id), pair in grouped.items():
        if "possible" not in pair or "impossible" not in pair:
            continue
        gap = float(pair["impossible"]["aggregated_surprise_localized"]) - float(
            pair["possible"]["aggregated_surprise_localized"]
        )
        pairs.append((abs(gap), set_id, pair_id, pair["possible"], pair["impossible"]))
    pairs.sort(reverse=True)
    return pairs


def build_groups(args):
    if args.source == "intphys2-debug":
        groups = discover_intphys2_debug(args.data_root, max_sets=args.max_sets)
        read_windows = lambda row: read_video_mp4_windows(
            row["local_path"], args.frame_step, args.stride, args.max_windows, args.decode_short_side
        )
    else:
        groups = discover_original_intphys(
            args.data_root,
            args.blocks,
            max_sets=args.max_sets,
            max_sets_per_block=args.max_sets_per_block,
        )
        read_windows = lambda row: read_original_intphys_variant_windows(
            row["local_path"], args.frame_step, args.stride, args.max_windows, args.decode_short_side
        )
    return groups, read_windows


def find_dataset_row(groups, csv_row):
    for row in groups[csv_row["set_id"]]:
        label = label_from_type(row["type"])
        movie_id = row.get("name") or Path(row["local_path"]).stem
        if label == csv_row["label"] and movie_id == csv_row["movie_id"]:
            return row
    for row in groups[csv_row["set_id"]]:
        if label_from_type(row["type"]) == csv_row["label"] and variant_pair_id(row["type"]) == csv_row["pair_id"]:
            return row
    raise KeyError(csv_row)


def diagnostic_flags_for_pair(pos_windows, imp_windows):
    flags = []
    for pos_window, imp_window in zip(pos_windows, imp_windows):
        active = active_mask_for_targets(pos_window[CONTEXT_FRAMES:], imp_window[CONTEXT_FRAMES:])
        flags.append(bool(active.size > 0 and active.any()))
    return np.asarray(flags, dtype=bool)


def aggregate(scores, flags, mode):
    scores = np.asarray(scores, dtype=np.float64)
    selected = scores[flags]
    if len(selected) == 0:
        return float("nan")
    if mode == "max":
        return float(selected.max())
    if mode == "mean":
        return float(selected.mean())
    raise ValueError(mode)


def score_controls(engine, windows):
    normal, reversed_scores, shuffled = [], [], []
    for window in windows:
        context = window[:CONTEXT_FRAMES]
        target = window[CONTEXT_FRAMES:]
        normal_score, _ = engine.compute_surprise(context, target)
        reversed_score, _ = engine.compute_surprise(context, target[::-1].copy())
        shuffled_score, _ = engine.compute_surprise(context, target[SHUFFLE_ORDER].copy())
        normal.append(normal_score)
        reversed_scores.append(reversed_score)
        shuffled.append(shuffled_score)
    return {
        "normal": np.asarray(normal, dtype=np.float64),
        "reversed": np.asarray(reversed_scores, dtype=np.float64),
        "shuffled": np.asarray(shuffled, dtype=np.float64),
    }


def save_control_dashboard(set_id, pair_id, label, windows, scores, flags, out_path):
    diagnostic_indices = np.where(flags)[0]
    if len(diagnostic_indices):
        gaps = np.maximum(scores["reversed"], scores["shuffled"]) - scores["normal"]
        window_idx = int(diagnostic_indices[np.argmax(gaps[diagnostic_indices])])
    else:
        window_idx = int(np.argmax(scores["normal"]))
    window = windows[window_idx]
    target = window[CONTEXT_FRAMES:]
    reversed_target = target[::-1]
    shuffled_target = target[SHUFFLE_ORDER]

    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(4, 8, height_ratios=[1.0, 1.0, 1.0, 1.2])
    frame_indices = np.linspace(0, FRAMES_PER_CLIP - 1, 8).round().astype(int)
    for col, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(window[frame_idx])
        ax.set_title(f"normal f{frame_idx}", fontsize=8)
        ax.axis("off")
    for col in range(8):
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(reversed_target[col])
        ax.set_title(f"reversed t{col}", fontsize=8)
        ax.axis("off")
        ax = fig.add_subplot(grid[2, col])
        ax.imshow(shuffled_target[col])
        ax.set_title(f"shuffled t{col}", fontsize=8)
        ax.axis("off")

    ax_trace = fig.add_subplot(grid[3, :5])
    x = np.arange(len(scores["normal"]))
    ax_trace.plot(x, scores["normal"], marker="o", label="normal", color="#2b6cb0")
    ax_trace.plot(x, scores["reversed"], marker="o", label="reversed", color="#c53030")
    ax_trace.plot(x, scores["shuffled"], marker="o", label="shuffled", color="#2f855a")
    ax_trace.scatter(x[flags], scores["normal"][flags], s=90, facecolors="none", edgecolors="#111827", label="diagnostic")
    ax_trace.axvline(window_idx, color="#111827", linestyle="--", linewidth=1)
    ax_trace.set_title("Engine-control per-window surprise")
    ax_trace.set_xlabel("window")
    ax_trace.set_ylabel("surprise")
    ax_trace.legend(ncol=2, fontsize=8)

    ax_bar = fig.add_subplot(grid[3, 5:])
    values = [aggregate(scores[name], flags, "max") for name in ("normal", "reversed", "shuffled")]
    ax_bar.bar(["normal", "reversed", "shuffled"], values, color=["#2b6cb0", "#c53030", "#2f855a"])
    ax_bar.set_title("Diagnostic-window max")
    ax_bar.set_ylabel("surprise")
    for i, value in enumerate(values):
        ax_bar.text(i, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        f"Engine control: {set_id} pair {pair_id} {label} | diagnostic windows={int(flags.sum())}/{len(flags)}",
        fontsize=13,
    )
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
    parser.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    rows = read_rows(args.csv)
    groups, read_windows = build_groups(args)
    engine = SurpriseEngine(checkpoint_path=args.checkpoint)
    selected = pair_rows(rows)[: args.limit]
    out_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {engine.device}")
    print(f"Precision: {engine.precision}")

    for _, set_id, pair_id, pos_row, imp_row in selected:
        pos_data = find_dataset_row(groups, pos_row)
        imp_data = find_dataset_row(groups, imp_row)
        pos_windows, _, _, _ = read_windows(pos_data)
        imp_windows, _, _, _ = read_windows(imp_data)
        flags = diagnostic_flags_for_pair(pos_windows, imp_windows)
        for label, row, windows in (("possible", pos_row, pos_windows), ("impossible", imp_row, imp_windows)):
            scores = score_controls(engine, windows)
            values = {name: aggregate(scores[name], flags, "max") for name in ("normal", "reversed", "shuffled")}
            print(
                f"{set_id} pair={pair_id} {label}: "
                f"diagnostic_windows={int(flags.sum())}/{len(flags)} "
                f"normal_diag_max={values['normal']:.6f} "
                f"reversed_diag_max={values['reversed']:.6f} "
                f"shuffled_diag_max={values['shuffled']:.6f}"
            )
            out_path = out_dir / f"intphys_engine_control_{set_id.replace(':', '_')}_p{pair_id}_{label}.png"
            save_control_dashboard(set_id, pair_id, label, windows, scores, flags, out_path)
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
