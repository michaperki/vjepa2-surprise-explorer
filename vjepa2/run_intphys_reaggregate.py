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
    active_mask_for_targets,
    discover_intphys2_debug,
    discover_original_intphys,
    label_from_type,
    read_original_intphys_variant_windows,
    read_video_mp4_windows,
    variant_pair_id,
)


def read_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def parse_scores(value):
    if not value:
        return np.asarray([], dtype=np.float64)
    return np.asarray([float(x) for x in value.split()], dtype=np.float64)


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
    for row in groups[csv_row["set_id"]]:
        label = label_from_type(row["type"])
        movie_id = row.get("name") or Path(row["local_path"]).stem
        if label == csv_row["label"] and movie_id == csv_row["movie_id"]:
            return row
    for row in groups[csv_row["set_id"]]:
        if label_from_type(row["type"]) == csv_row["label"] and variant_pair_id(row["type"]) == csv_row["pair_id"]:
            return row
    raise KeyError(f"missing row for {csv_row['set_id']} {csv_row['pair_id']} {csv_row['label']}")


def pair_csv_rows(rows):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["set_id"], row["pair_id"])][row["label"]] = row
    return [
        {"set_id": set_id, "pair_id": pair_id, "possible": pair["possible"], "impossible": pair["impossible"]}
        for (set_id, pair_id), pair in grouped.items()
        if "possible" in pair and "impossible" in pair
    ]


def diagnostic_flags_for_pair(pos_windows, imp_windows):
    flags = []
    active_counts = []
    for pos_window, imp_window in zip(pos_windows, imp_windows):
        active = active_mask_for_targets(pos_window[CONTEXT_FRAMES:], imp_window[CONTEXT_FRAMES:])
        flags.append(bool(active.size > 0 and active.any()))
        active_counts.append(int(active.sum()) if active.size > 0 else 0)
    return np.asarray(flags, dtype=bool), active_counts


def aggregate(scores, flags, mode):
    selected = scores[flags]
    if len(selected) == 0:
        return float("nan")
    if mode == "max":
        return float(selected.max())
    if mode == "mean":
        return float(selected.mean())
    raise ValueError(mode)


def accuracy(records, field):
    measured = [r for r in records if not r["not_measurable"]]
    if not measured:
        return float("nan"), 0, 0
    correct = sum(r[f"impossible_{field}"] > r[f"possible_{field}"] for r in measured)
    ties = sum(abs(r[f"impossible_{field}"] - r[f"possible_{field}"]) < 1e-12 for r in measured)
    return correct / len(measured), len(measured), ties


def old_accuracy(records):
    correct = sum(r["impossible_old"] > r["possible_old"] for r in records)
    ties = sum(abs(r["impossible_old"] - r["possible_old"]) < 1e-12 for r in records)
    return correct / len(records), len(records), ties


def old_localized_accuracy(records):
    correct = sum(r["impossible_old_localized"] > r["possible_old_localized"] for r in records)
    ties = sum(abs(r["impossible_old_localized"] - r["possible_old_localized"]) < 1e-12 for r in records)
    return correct / len(records), len(records), ties


def save_before_after(old, diag_max, diag_mean, not_measurable, out_path):
    labels = ["old localized all-window max", "diagnostic max", "diagnostic mean", "not measurable"]
    values = [old[0], diag_max[0], diag_mean[0], not_measurable / old[1] if old[1] else 0.0]
    counts = [old[1], diag_max[1], diag_mean[1], not_measurable]
    colors = ["#a0aec0", "#2b6cb0", "#2f855a", "#c53030"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(labels, values, color=colors)
    ax.axhline(0.5, color="#718096", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy / fraction")
    ax.set_title("IntPhys O1 before/after diagnostic-window aggregation")
    for bar, value, count in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}\nn={count}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
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
    args = parser.parse_args()

    rows = read_rows(args.csv)
    groups, read_windows = build_groups(args)
    records = []

    for pair in pair_csv_rows(rows):
        pos_row = pair["possible"]
        imp_row = pair["impossible"]
        pos_data = find_dataset_row(groups, pos_row)
        imp_data = find_dataset_row(groups, imp_row)
        pos_windows, _, _, _ = read_windows(pos_data)
        imp_windows, _, _, _ = read_windows(imp_data)
        flags, active_counts = diagnostic_flags_for_pair(pos_windows, imp_windows)

        pos_scores = parse_scores(pos_row["window_surprises_localized"])
        imp_scores = parse_scores(imp_row["window_surprises_localized"])
        old_pos = float(pos_row["aggregated_surprise_all"])
        old_imp = float(imp_row["aggregated_surprise_all"])
        old_local_pos = float(pos_row["aggregated_surprise_localized"])
        old_local_imp = float(imp_row["aggregated_surprise_localized"])

        records.append(
            {
                "set_id": pair["set_id"],
                "pair_id": pair["pair_id"],
                "not_measurable": not flags.any(),
                "diagnostic_windows": int(flags.sum()),
                "active_tokens_mean": float(np.mean([c for c in active_counts if c > 0])) if flags.any() else 0.0,
                "possible_old": old_pos,
                "impossible_old": old_imp,
                "possible_old_localized": old_local_pos,
                "impossible_old_localized": old_local_imp,
                "possible_diag_max": aggregate(pos_scores, flags, "max"),
                "impossible_diag_max": aggregate(imp_scores, flags, "max"),
                "possible_diag_mean": aggregate(pos_scores, flags, "mean"),
                "impossible_diag_mean": aggregate(imp_scores, flags, "mean"),
            }
        )

    old = old_localized_accuracy(records)
    old_all = old_accuracy(records)
    diag_max = accuracy(records, "diag_max")
    diag_mean = accuracy(records, "diag_mean")
    not_measurable = sum(r["not_measurable"] for r in records)
    non_tie_records = [
        r for r in records if abs(r["impossible_old_localized"] - r["possible_old_localized"]) >= 1e-12
    ]
    old_non_tie_correct = sum(r["impossible_old_localized"] > r["possible_old_localized"] for r in non_tie_records)

    out_path = Path(args.figures_dir) / "intphys_probe_diagnostic_before_after.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_before_after(old, diag_max, diag_mean, not_measurable, out_path)

    print("Before / after diagnostic-window reaggregation")
    print(f"old localized all-window max: accuracy={old[0]:.4f} n={old[1]} ties={old[2]}")
    print(f"old all-token all-window max: accuracy={old_all[0]:.4f} n={old_all[1]} ties={old_all[2]}")
    print(
        f"old non-tie-only reference: correct={old_non_tie_correct}/{len(non_tie_records)} "
        f"accuracy={old_non_tie_correct / len(non_tie_records):.4f}"
    )
    print(f"diagnostic max: accuracy={diag_max[0]:.4f} measurable_n={diag_max[1]} ties={diag_max[2]}")
    print(f"diagnostic mean: accuracy={diag_mean[0]:.4f} measurable_n={diag_mean[1]} ties={diag_mean[2]}")
    print(f"not_measurable pairs: {not_measurable}/{len(records)}")
    print(f"figure: {out_path}")


if __name__ == "__main__":
    main()
