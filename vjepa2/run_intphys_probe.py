import argparse
import csv
import gc
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from huggingface_hub import hf_hub_download

from run_physics_probe_v2 import active_token_mask
from surprise_engine import CONTEXT_FRAMES, FRAMES_PER_CLIP, SurpriseEngine


INTPHYS2_REPO = "facebook/IntPhys2"
INTPHYS2_REVISION = "main"
INTPHYS2_DEBUG_ROWS = 60
INTPHYS_ORIGINAL_DEV_SIZE = "3 GB archive; 30 scenes per block O1/O2/O3"


def load_hf_debug_metadata(max_sets=None):
    from datasets import load_dataset

    rows = [dict(row) for row in load_dataset(INTPHYS2_REPO, split="Debug")]
    groups = defaultdict(list)
    for row in rows:
        groups[row["SceneIndex"]].append(row)
    ordered_ids = sorted(groups)
    if max_sets is not None:
        ordered_ids = ordered_ids[:max_sets]
    return {set_id: sorted(groups[set_id], key=lambda r: r["type"]) for set_id in ordered_ids}


def prepare_intphys2_debug(data_dir, max_sets=None):
    data_dir = Path(data_dir)
    videos_dir = data_dir / "Debug"
    videos_dir.mkdir(parents=True, exist_ok=True)
    groups = load_hf_debug_metadata(max_sets=max_sets)
    for rows in groups.values():
        for row in rows:
            local_path = videos_dir / Path(row["file_name"]).name
            if local_path.exists():
                row["local_path"] = str(local_path)
                continue
            cached = hf_hub_download(
                repo_id=INTPHYS2_REPO,
                repo_type="dataset",
                filename=f"Debug/{row['file_name']}",
                revision=INTPHYS2_REVISION,
            )
            local_path.write_bytes(Path(cached).read_bytes())
            row["local_path"] = str(local_path)
    metadata_path = data_dir / "debug_metadata.json"
    metadata_path.write_text(json.dumps(groups, indent=2))
    return groups


def discover_intphys2_debug(data_dir, max_sets=None):
    data_dir = Path(data_dir)
    metadata_path = data_dir / "debug_metadata.json"
    if metadata_path.exists():
        groups = json.loads(metadata_path.read_text())
    else:
        groups = load_hf_debug_metadata(max_sets=max_sets)
    for rows in groups.values():
        for row in rows:
            row.setdefault("local_path", str(data_dir / "Debug" / Path(row["file_name"]).name))
    ordered_ids = sorted(groups)
    if max_sets is not None:
        ordered_ids = ordered_ids[:max_sets]
    return {set_id: groups[set_id] for set_id in ordered_ids}


def discover_original_intphys(data_root, blocks, max_sets=None, max_sets_per_block=None):
    root = Path(data_root)
    groups = {}
    for block in blocks:
        block_count = 0
        block_root = root / block
        if not block_root.exists():
            alt = root / "dev" / block
            block_root = alt if alt.exists() else block_root
        if not block_root.exists():
            continue
        for scene_dir in sorted(p for p in block_root.iterdir() if p.is_dir()):
            set_id = f"{block}:{scene_dir.name}"
            rows = []
            for possibility in ("1", "2", "3", "4"):
                variant_dir = scene_dir / possibility
                scene_frames = variant_dir / "scene"
                if not scene_frames.exists():
                    continue
                status_path = variant_dir / "status.json"
                is_possible = possibility in ("1", "2")
                status = {}
                if status_path.exists():
                    status = json.loads(status_path.read_text())
                    is_possible = bool(status.get("header", {}).get("is_possible", is_possible))
                rows.append(
                    {
                        "SceneIndex": set_id,
                        "type": f"{possibility}_{'Possible' if is_possible else 'Impossible'}",
                        "condition": block,
                        "property": block,
                        "occluder": infer_occlusion(status),
                        "local_path": str(variant_dir),
                    }
                )
            if rows:
                groups[set_id] = rows
                block_count += 1
            if max_sets_per_block is not None and block_count >= max_sets_per_block:
                break
            if max_sets is not None and len(groups) >= max_sets:
                return groups
    return groups


def infer_occlusion(status):
    text = json.dumps(status).lower()
    if "occlud" in text:
        if "false" in text and "true" not in text:
            return "visible"
        return "occluded"
    return "unknown"


def selected_window_starts(num_frames, frame_step, stride, max_windows=None):
    sampled_len = len(np.arange(0, num_frames, frame_step))
    if sampled_len < FRAMES_PER_CLIP:
        return [], sampled_len
    starts = list(range(0, sampled_len - FRAMES_PER_CLIP + 1, stride))
    if not starts:
        starts = [0]
    if max_windows is not None and len(starts) > max_windows:
        if max_windows == 1:
            starts = [starts[len(starts) // 2]]
        else:
            idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
            starts = [starts[i] for i in idx]
    return starts, sampled_len


def needed_frame_indices(num_frames, frame_step, stride, max_windows=None):
    starts, sampled_len = selected_window_starts(
        num_frames=num_frames,
        frame_step=frame_step,
        stride=stride,
        max_windows=max_windows,
    )
    sampled_indices = np.arange(0, num_frames, frame_step, dtype=np.int64)
    needed_sample_positions = []
    for start in starts:
        needed_sample_positions.extend(range(start, start + FRAMES_PER_CLIP))
    needed_sample_positions = sorted(set(needed_sample_positions))
    needed = sampled_indices[needed_sample_positions] if needed_sample_positions else np.asarray([], dtype=np.int64)
    return starts, sampled_len, needed.tolist()


def resize_frame(frame, decode_short_side):
    if decode_short_side is None:
        return frame
    image = Image.fromarray(frame)
    width, height = image.size
    short = min(width, height)
    if short == decode_short_side:
        return frame
    scale = decode_short_side / short
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(image.resize(new_size, Image.BILINEAR))


def make_windows_from_sparse_frames(frame_by_index, num_frames, frame_step, stride, max_windows):
    starts, _, _ = needed_frame_indices(
        num_frames=num_frames,
        frame_step=frame_step,
        stride=stride,
        max_windows=max_windows,
    )
    sampled_indices = np.arange(0, num_frames, frame_step, dtype=np.int64)
    windows = []
    for start in starts:
        original_indices = sampled_indices[start : start + FRAMES_PER_CLIP]
        windows.append(np.stack([frame_by_index[int(idx)] for idx in original_indices], axis=0))
    return windows


def read_video_mp4_windows(path, frame_step, stride, max_windows, decode_short_side):
    reader_kwargs = {"num_threads": -1, "ctx": cpu(0)}
    if decode_short_side is not None:
        # IntPhys/IntPhys2 stimuli are square; this avoids decoding full-res frames.
        reader_kwargs.update({"width": decode_short_side, "height": decode_short_side})
    try:
        vr = VideoReader(str(path), **reader_kwargs)
        resize_after_decode = False
    except TypeError:
        reader_kwargs.pop("width", None)
        reader_kwargs.pop("height", None)
        vr = VideoReader(str(path), **reader_kwargs)
        resize_after_decode = True

    num_frames = len(vr)
    starts, _, needed = needed_frame_indices(
        num_frames=num_frames,
        frame_step=frame_step,
        stride=stride,
        max_windows=max_windows,
    )
    if not starts:
        return [], np.empty((0, 0, 0, 3), dtype=np.uint8), np.empty((0, 0, 0, 3), dtype=np.uint8), {
            "num_frames": num_frames,
            "frames_read": 0,
            "resolution": "n/a",
        }

    decoded = vr.get_batch(np.asarray(needed, dtype=np.int64)).asnumpy()
    if resize_after_decode and decode_short_side is not None:
        decoded = np.stack([resize_frame(frame, decode_short_side) for frame in decoded], axis=0)
    frame_by_index = {idx: frame for idx, frame in zip(needed, decoded)}
    windows = make_windows_from_sparse_frames(frame_by_index, num_frames, frame_step, stride, max_windows)
    motion_frames = np.stack([frame_by_index[idx] for idx in needed], axis=0)
    example_frames = motion_frames[
        np.linspace(0, len(motion_frames) - 1, min(8, len(motion_frames))).round().astype(int)
    ]
    h, w = motion_frames.shape[1:3]
    return windows, motion_frames, example_frames, {
        "num_frames": num_frames,
        "frames_read": len(needed),
        "resolution": f"{w}x{h}",
    }


def read_original_intphys_variant_windows(path, frame_step, stride, max_windows, decode_short_side):
    frame_dir = Path(path) / "scene"
    frame_paths = sorted(frame_dir.iterdir())
    num_frames = len(frame_paths)
    starts, _, needed = needed_frame_indices(
        num_frames=num_frames,
        frame_step=frame_step,
        stride=stride,
        max_windows=max_windows,
    )
    if not starts:
        return [], np.empty((0, 0, 0, 3), dtype=np.uint8), np.empty((0, 0, 0, 3), dtype=np.uint8), {
            "num_frames": num_frames,
            "frames_read": 0,
            "resolution": "n/a",
        }
    frames = []
    for idx in needed:
        frame = np.asarray(Image.open(frame_paths[idx]).convert("RGB"))
        frames.append(resize_frame(frame, decode_short_side))
    frame_by_index = {idx: frame for idx, frame in zip(needed, frames)}
    windows = make_windows_from_sparse_frames(frame_by_index, num_frames, frame_step, stride, max_windows)
    motion_frames = np.stack([frame_by_index[idx] for idx in needed], axis=0)
    example_frames = motion_frames[
        np.linspace(0, len(motion_frames) - 1, min(8, len(motion_frames))).round().astype(int)
    ]
    h, w = motion_frames.shape[1:3]
    return windows, motion_frames, example_frames, {
        "num_frames": num_frames,
        "frames_read": len(needed),
        "resolution": f"{w}x{h}",
    }


def sample_windows(frames, frame_step, stride, max_windows=None):
    sampled = frames[::frame_step]
    if len(sampled) < FRAMES_PER_CLIP:
        return []
    starts = list(range(0, len(sampled) - FRAMES_PER_CLIP + 1, stride))
    if not starts:
        starts = [0]
    if max_windows is not None and len(starts) > max_windows:
        if max_windows == 1:
            starts = [starts[len(starts) // 2]]
        else:
            idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
            starts = [starts[i] for i in idx]
    return [sampled[start : start + FRAMES_PER_CLIP] for start in starts]


def motion_energy(frames):
    arr = frames.astype(np.float32) / 255.0
    if len(arr) < 2:
        return 0.0
    return float(np.mean(np.abs(arr[1:] - arr[:-1])))


def score_movie_windows(engine, windows, aggregate):
    if not windows:
        raise ValueError("movie has fewer sampled frames than one 16-frame window")
    window_scores = []
    for window in windows:
        scalar, _ = engine.compute_surprise(window[:CONTEXT_FRAMES], window[CONTEXT_FRAMES:])
        window_scores.append(scalar)
    window_scores = np.asarray(window_scores, dtype=np.float64)
    if aggregate == "max":
        aggregate_score = float(window_scores.max())
    elif aggregate == "mean":
        aggregate_score = float(window_scores.mean())
    else:
        raise ValueError("aggregate must be max or mean")
    return aggregate_score, window_scores


def score_window_maps(engine, context, targets):
    outputs = engine.compute_surprises(context, targets)
    return [(float(score), token_map) for score, token_map in outputs]


def aggregate_scores(scores, aggregate):
    scores = np.asarray(scores, dtype=np.float64)
    if len(scores) == 0:
        return float("nan")
    if aggregate == "max":
        return float(scores.max())
    if aggregate == "mean":
        return float(scores.mean())
    raise ValueError("aggregate must be max or mean")


def diagnostic_aggregate(scores, diagnostic_flags, mode):
    scores = np.asarray(scores, dtype=np.float64)
    flags = np.asarray(diagnostic_flags, dtype=bool)
    selected = scores[flags]
    if len(selected) == 0:
        return float("nan")
    if mode == "max":
        return float(selected.max())
    if mode == "mean":
        return float(selected.mean())
    raise ValueError("mode must be max or mean")


def label_from_type(type_name):
    lower = type_name.lower()
    if "possible" in lower and "impossible" not in lower:
        return "possible"
    if "impossible" in lower:
        return "impossible"
    raise ValueError(f"cannot infer label from type={type_name}")


def variant_pair_id(type_name):
    prefix = str(type_name).split("_", 1)[0]
    return prefix if prefix.isdigit() else None


def condition_from_row(row):
    return row.get("condition") or row.get("Difficulty") or "unknown"


def property_from_row(row):
    return row.get("property") or row.get("condition") or row.get("game_name") or "unknown"


def active_mask_for_targets(pos_target, imp_target):
    if not np.any(pos_target != imp_target):
        return np.zeros(0, dtype=bool)
    return active_token_mask({"possible": pos_target, "impossible": imp_target, "linear": pos_target})


def context_distance(pos_windows, imp_windows):
    if len(pos_windows) != len(imp_windows):
        return float("inf")
    if not pos_windows:
        return float("inf")
    diffs = []
    for pos_window, imp_window in zip(pos_windows, imp_windows):
        diffs.append(np.mean(np.abs(pos_window[:CONTEXT_FRAMES].astype(np.int16) - imp_window[:CONTEXT_FRAMES].astype(np.int16))))
    return float(np.mean(diffs))


def match_pairs(movie_items):
    positives = [item for item in movie_items if item["label"] == "possible"]
    negatives = [item for item in movie_items if item["label"] == "impossible"]
    pairs = []
    used_negatives = set()

    positives_by_pid = defaultdict(list)
    negatives_by_pid = defaultdict(list)
    for item in positives:
        positives_by_pid[item["pair_id"]].append(item)
    for item in negatives:
        negatives_by_pid[item["pair_id"]].append(item)

    for pair_id, pos_items in positives_by_pid.items():
        neg_items = negatives_by_pid.get(pair_id, [])
        while pos_items and neg_items:
            pos = pos_items.pop(0)
            neg = neg_items.pop(0)
            pairs.append((pos, neg))
            used_negatives.add(id(neg))

    remaining_pos = [item for item in positives if not any(id(item) == id(p[0]) for p in pairs)]
    remaining_neg = [item for item in negatives if id(item) not in used_negatives]
    for pos in remaining_pos:
        if not remaining_neg:
            break
        distances = [context_distance(pos["windows"], neg["windows"]) for neg in remaining_neg]
        neg_index = int(np.argmin(distances))
        neg = remaining_neg.pop(neg_index)
        pairs.append((pos, neg))
    return pairs


def pair_accuracy(pair_rows, metric="localized", subset=None):
    rows = pair_rows
    if subset is None:
        rows = [row for row in rows if not row.get("not_measurable", False)]
    elif subset == "differentiable":
        rows = [row for row in rows if row["differentiable"]]
    elif subset in {"undifferentiable", "not_measurable"}:
        rows = [row for row in rows if row.get("not_measurable", False)]
    if not rows:
        return float("nan"), 0
    if metric == "localized":
        possible_key = "possible_localized"
        impossible_key = "impossible_localized"
    elif metric == "diagnostic_mean":
        possible_key = "possible_diagnostic_mean"
        impossible_key = "impossible_diagnostic_mean"
    else:
        possible_key = "possible_all"
        impossible_key = "impossible_all"
    correct = sum(row[impossible_key] > row[possible_key] for row in rows)
    return correct / len(rows), len(rows)


def breakdown_pair_accuracy(pair_rows, key, metric="localized"):
    buckets = defaultdict(list)
    for row in pair_rows:
        buckets[row.get(key, "unknown")].append(row)
    return {name: pair_accuracy(rows, metric=metric) for name, rows in buckets.items()}


CSV_FIELDS = [
        "movie_id",
        "set_id",
        "pair_id",
        "label",
        "aggregated_surprise",
        "aggregated_surprise_all",
        "aggregated_surprise_localized",
        "aggregated_surprise_diagnostic_max",
        "aggregated_surprise_diagnostic_mean",
        "aggregated_surprise_old_all_window_max",
        "motion_energy",
        "condition",
        "property",
        "source",
        "num_windows",
        "differentiable_windows",
        "active_tokens_mean",
        "window_surprises",
        "window_surprises_all",
        "window_surprises_localized",
        "diagnostic_window_flags",
        "matched_movie_id",
        "differentiable",
        "not_measurable",
    ]


def open_incremental_csv(path):
    """Open the per-movie CSV, resuming if it already holds rows.

    Returns (handle, writer, done_ids). On a flaky GPU a long run can crash
    partway; re-running appends and skips movies already scored, so progress
    is never lost. The per-row flush means rows up to a crash survive.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if path.exists() and path.stat().st_size > 0:
        with path.open(newline="") as existing:
            for row in csv.DictReader(existing):
                # movie_id is only unique within a scene (1..4 for intphys-dev),
                # so key resume on (set_id, movie_id).
                if row.get("movie_id"):
                    done_ids.add((row.get("set_id", ""), row["movie_id"]))
    if done_ids:
        handle = path.open("a", newline="")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    else:
        handle = path.open("w", newline="")
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        handle.flush()
    return handle, writer, done_ids


def append_csv_row(writer, handle, row):
    writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
    handle.flush()


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def save_accuracy_figure(labels, values, counts, out_path, title="Localized VoE accuracy"):
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.4), 4.2))
    bars = ax.bar(labels, values, color="#2b6cb0")
    ax.axhline(0.5, color="#718096", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("VoE relative accuracy")
    ax.set_title(title)
    for bar, value, count in zip(bars, values, counts):
        text = f"{value:.2f}\nn={count}" if not np.isnan(value) else f"n/a\nn={count}"
        ax.text(bar.get_x() + bar.get_width() / 2, 0 if np.isnan(value) else value, text, ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_localized_vs_all_figure(localized, all_token, out_path):
    labels = ["overall", "differentiable", "undifferentiable"]
    local_values = [localized[name][0] for name in labels]
    all_values = [all_token[name][0] for name in labels]
    counts = [localized[name][1] for name in labels]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width / 2, local_values, width, label="localized", color="#2b6cb0")
    ax.bar(x + width / 2, all_values, width, label="all tokens", color="#a0aec0")
    ax.axhline(0.5, color="#718096", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xticks(x, [f"{label}\nn={count}" for label, count in zip(labels, counts)])
    ax.set_ylabel("VoE relative accuracy")
    ax.set_title("Localized vs all-token VoE accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_motion_scatter(rows, out_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = {"possible": "#2b6cb0", "impossible": "#c53030"}
    for label in ("possible", "impossible"):
        xs = [r["motion_energy"] for r in rows if r["label"] == label]
        ys = [r["aggregated_surprise_localized"] for r in rows if r["label"] == label]
        ax.scatter(xs, ys, label=label, alpha=0.75, s=35, color=colors[label])
    ax.set_xlabel("mean abs frame-to-frame pixel difference")
    ax.set_ylabel("localized aggregated surprise")
    ax.set_title("Localized surprise vs low-level motion energy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_examples(examples, out_path):
    if not examples:
        return
    cols = 8
    rows = len(examples)
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(2.8, 2.4 * rows)))
    if rows == 1:
        axes = np.expand_dims(axes, 0)
    for row_idx, ex in enumerate(examples):
        frames = ex["frames"]
        idx = np.linspace(0, len(frames) - 1, cols).round().astype(int)
        for col, frame_idx in enumerate(idx):
            ax = axes[row_idx, col]
            ax.imshow(frames[frame_idx])
            ax.set_title(f"{ex['label']} {ex['score']:.3f}\nf{frame_idx}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Example movies: sampled frames with aggregated surprise", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(args, source_info, localized, all_token, by_block, by_property, csv_path, figures, num_movies):
    diff_n = localized["differentiable"][1]
    undiff_n = localized["undifferentiable"][1]
    lines = [
        "# Real benchmark surprise probe",
        "",
        "## Data slice",
        "",
        source_info,
        f"Movies scored: `{num_movies}`",
        "",
        "Original JEPA intuitive-physics repo survey: its bundled `data_intphys.tar.gz` is precomputed model surprises/performance files, not raw videos. The official raw IntPhys dev archive is 3 GB, with 30 scenes per block and folders `dev/O1|O2|O3/<scene>/<1..4>/scene/*.png` plus `status.json` labels. This runner can consume that local structure, but does not auto-download the 3 GB archive.",
        "",
        "## Protocol",
        "",
        f"Each movie is sampled into 16-frame windows with `frame_step={args.frame_step}` and `stride={args.stride}` over sampled frames. Each window is scored by `SurpriseEngine`: first 8 frames are context and last 8 are target, using the existing V-JEPA 2 ViT-L predictor, target layer norm, and mean absolute embedding error. Movie surprise is aggregated by `{args.aggregate}` over window scores. This mirrors the JEPA intuitive-physics protocol's sliding-window losses and relative possible-vs-impossible comparison, while using our fixed ViT-L engine.",
        "",
        "Localized metric: within each matched possible/impossible pair and aligned target window, the active-token mask is built from target-frame pixel differences using the same 16x16-per-slot token-grid localization as `run_physics_probe_v2.py`. Empty active masks are marked `undifferentiable` and use all-token values as the localized fallback only for CSV completeness; accuracy is reported separately for differentiable and undifferentiable subsets.",
        "",
        "Model: this is the ViT-L checkpoint, run deliberately (it fits the hardware). The question we ask is whether *this* model's surprise separates possible from impossible, not whether a larger one would. We make no claim that ViT-Huge would close the gap — that is an untested assumption, not a result; testing it is a separate, open experiment.",
        "",
        "## Accuracy",
        "",
        f"Localized overall relative VoE accuracy: `{localized['overall'][0]:.4f}` over `{localized['overall'][1]}` matched pairs.",
        f"Differentiable pairs: `{diff_n}`; undifferentiable pairs: `{undiff_n}`.",
        f"Localized differentiable accuracy: `{localized['differentiable'][0]:.4f}` over `{localized['differentiable'][1]}` pairs.",
        f"Localized undifferentiable accuracy: `{localized['undifferentiable'][0]:.4f}` over `{localized['undifferentiable'][1]}` pairs.",
        "",
        f"All-token overall accuracy: `{all_token['overall'][0]:.4f}` over `{all_token['overall'][1]}` matched pairs.",
        f"All-token differentiable accuracy: `{all_token['differentiable'][0]:.4f}` over `{all_token['differentiable'][1]}` pairs.",
        f"All-token undifferentiable accuracy: `{all_token['undifferentiable'][0]:.4f}` over `{all_token['undifferentiable'][1]}` pairs.",
        "",
        "Localized by block/condition:",
    ]
    for name, (acc, n) in by_block.items():
        lines.append(f"- `{name}`: `{acc:.4f}` over `{n}` sets")
    lines.append("")
    lines.append("Localized by physical property:")
    for name, (acc, n) in by_property.items():
        lines.append(f"- `{name}`: `{acc:.4f}` over `{n}` sets")
    lines += [
        "",
        f"CSV: `{csv_path}`",
        "",
        "Figures:",
    ]
    for label, path in figures:
        lines.append(f"- `{path}`: {label}")
    Path(args.report).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["intphys2-debug", "intphys-dev"], default="intphys2-debug")
    parser.add_argument("--data-root", default="data/intphys2_debug")
    parser.add_argument("--download-intphys2-debug", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    parser.add_argument(
        "--weights-dtype",
        choices=["fp32", "bf16", "fp16"],
        default="fp32",
        help="Resident model weight precision; bf16 ~halves GPU memory (compute is bf16 either way)",
    )
    parser.add_argument("--blocks", nargs="+", default=["O1", "O2", "O3"])
    parser.add_argument("--max-sets", type=int, default=15)
    parser.add_argument("--max-sets-per-block", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--frame-step", type=int, default=2)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--aggregate", choices=["max", "mean"], default="max")
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--csv", default="outputs/intphys_probe_movies.csv")
    parser.add_argument("--report", default="REAL_BENCHMARK.md")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--decode-short-side", type=int, default=320)
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "intphys2-debug":
        if args.download_intphys2_debug:
            groups = prepare_intphys2_debug(args.data_root, max_sets=args.max_sets)
        else:
            groups = discover_intphys2_debug(args.data_root, max_sets=args.max_sets)
        source_info = (
            f"Source: `facebook/IntPhys2` Debug split. Full Debug split is 5 scenes / 60 videos; "
            f"this run uses up to `{args.max_sets}` matched SceneIndex groups from `{args.data_root}`. "
            "Folder after download: `data/intphys2_debug/Debug/*.mp4` plus `debug_metadata.json`."
        )
        read_windows = lambda row: read_video_mp4_windows(
            row["local_path"],
            frame_step=args.frame_step,
            stride=args.stride,
            max_windows=args.max_windows,
            decode_short_side=args.decode_short_side,
        )
        source_name = "intphys2-debug"
    else:
        groups = discover_original_intphys(
            args.data_root,
            blocks=args.blocks,
            max_sets=args.max_sets,
            max_sets_per_block=args.max_sets_per_block,
        )
        source_info = (
            f"Source: original IntPhys dev local folder `{args.data_root}`. "
            f"Official dev size is {INTPHYS_ORIGINAL_DEV_SIZE}; this run uses blocks `{','.join(args.blocks)}` "
            f"and up to `{args.max_sets}` matched scenes."
        )
        read_windows = lambda row: read_original_intphys_variant_windows(
            row["local_path"],
            frame_step=args.frame_step,
            stride=args.stride,
            max_windows=args.max_windows,
            decode_short_side=args.decode_short_side,
        )
        source_name = "intphys-dev"

    if not groups:
        raise SystemExit(
            "No dataset groups found. For the small runnable slice, run with "
            "`--source intphys2-debug --download-intphys2-debug`; for original IntPhys, pass `--data-root` "
            "pointing at an extracted dev folder."
        )

    missing = [
        row["local_path"]
        for rows in groups.values()
        for row in rows
        if not Path(row["local_path"]).exists()
    ]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} video paths. For IntPhys2 Debug, rerun with `--download-intphys2-debug`. "
            f"First missing: {missing[0]}"
        )

    if args.prepare_only:
        print(f"Prepared source `{source_name}` with {len(groups)} matched groups.")
        print(f"Data root: {args.data_root}")
        return

    _wdtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.weights_dtype]
    engine = SurpriseEngine(checkpoint_path=args.checkpoint, weights_dtype=_wdtype)
    print(f"Device: {engine.device}")
    print(f"Precision: {engine.precision}")
    print(f"Source: {source_name}")
    print(f"Matched groups: {len(groups)}")

    rows_out = []
    pair_rows = []
    examples = []
    movie_count = 0
    csv_path = Path(args.csv)
    csv_handle, csv_writer, done_ids = open_incremental_csv(csv_path)
    resumed = bool(done_ids)
    if resumed:
        print(f"Resuming: {len(done_ids)} movies already scored in {csv_path}; skipping those.")
    try:
      for set_id, movie_rows in groups.items():
        group_ids = [(set_id, (row.get("name") or Path(row["local_path"]).stem)) for row in movie_rows]
        if done_ids and all(mid in done_ids for mid in group_ids):
            continue  # whole scene already scored in a previous run; skip decode + GPU
        movie_items = []
        for row in movie_rows:
            label = label_from_type(row["type"])
            movie_id = row.get("name") or Path(row["local_path"]).stem
            windows, motion_frames, example_frames, footprint = read_windows(row)
            energy = motion_energy(motion_frames)
            movie_items.append({
                "row": row,
                "movie_id": movie_id,
                "set_id": set_id,
                "pair_id": variant_pair_id(row["type"]),
                "label": label,
                "motion_energy": energy,
                "condition": condition_from_row(row),
                "property": property_from_row(row),
                "source": source_name,
                "windows": windows,
                "footprint": footprint,
            })
            if len(examples) < 2:
                examples.append({
                    "frames": example_frames.copy(),
                    "label": label,
                    "score": float("nan"),
                    "movie_id": movie_id,
                })
            print(
                f"{set_id} {movie_id} {label} loaded "
                f"motion={energy:.6f} windows={len(windows)} "
                f"read={footprint['frames_read']}/{footprint['num_frames']} frames "
                f"resolution={footprint['resolution']}"
            )
            del motion_frames, example_frames
            gc.collect()
            movie_count += 1
            if movie_count % 4 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        for pos_item, imp_item in match_pairs(movie_items):
            if (set_id, pos_item["movie_id"]) in done_ids and (set_id, imp_item["movie_id"]) in done_ids:
                continue  # already scored in a previous (crashed) run
            num_windows = min(len(pos_item["windows"]), len(imp_item["windows"]))
            pos_all_scores = []
            imp_all_scores = []
            pos_local_scores = []
            imp_local_scores = []
            active_counts = []
            diagnostic_flags = []
            differentiable_windows = 0
            for window_idx in range(num_windows):
                pos_window = pos_item["windows"][window_idx]
                imp_window = imp_item["windows"][window_idx]
                pos_context = pos_window[:CONTEXT_FRAMES]
                imp_context = imp_window[:CONTEXT_FRAMES]
                pos_target = pos_window[CONTEXT_FRAMES:]
                imp_target = imp_window[CONTEXT_FRAMES:]

                pos_all, pos_map = engine.compute_surprise(pos_context, pos_target)
                imp_all, imp_map = engine.compute_surprise(imp_context, imp_target)
                pos_all_scores.append(pos_all)
                imp_all_scores.append(imp_all)

                active = active_mask_for_targets(pos_target, imp_target)
                if active.size == 0 or not active.any():
                    pos_local_scores.append(pos_all)
                    imp_local_scores.append(imp_all)
                    active_counts.append(0)
                    diagnostic_flags.append(False)
                else:
                    differentiable_windows += 1
                    active_counts.append(int(active.sum()))
                    diagnostic_flags.append(True)
                    pos_local_scores.append(float(np.mean(pos_map[active])))
                    imp_local_scores.append(float(np.mean(imp_map[active])))

            not_measurable = differentiable_windows == 0
            differentiable = not not_measurable
            pos_all_agg = aggregate_scores(pos_all_scores, args.aggregate)
            imp_all_agg = aggregate_scores(imp_all_scores, args.aggregate)
            pos_diag_max = diagnostic_aggregate(pos_local_scores, diagnostic_flags, "max")
            imp_diag_max = diagnostic_aggregate(imp_local_scores, diagnostic_flags, "max")
            pos_diag_mean = diagnostic_aggregate(pos_local_scores, diagnostic_flags, "mean")
            imp_diag_mean = diagnostic_aggregate(imp_local_scores, diagnostic_flags, "mean")
            pos_local_agg = pos_diag_max
            imp_local_agg = imp_diag_max
            active_tokens_mean = float(np.mean([c for c in active_counts if c > 0])) if differentiable else 0.0

            pair_id = pos_item["pair_id"] or imp_item["pair_id"] or f"match{len(pair_rows)}"
            pair_row = {
                "set_id": set_id,
                "pair_id": pair_id,
                "condition": pos_item["condition"],
                "property": pos_item["property"],
                "differentiable": differentiable,
                "not_measurable": not_measurable,
                "possible_all": pos_all_agg,
                "impossible_all": imp_all_agg,
                "possible_localized": pos_local_agg,
                "impossible_localized": imp_local_agg,
                "possible_diagnostic_mean": pos_diag_mean,
                "impossible_diagnostic_mean": imp_diag_mean,
            }
            pair_rows.append(pair_row)

            for item, all_agg, local_agg, all_scores, local_scores, matched_id in (
                (pos_item, pos_all_agg, pos_local_agg, pos_all_scores, pos_local_scores, imp_item["movie_id"]),
                (imp_item, imp_all_agg, imp_local_agg, imp_all_scores, imp_local_scores, pos_item["movie_id"]),
            ):
                out = {
                    "movie_id": item["movie_id"],
                    "set_id": item["set_id"],
                    "pair_id": pair_id,
                    "label": item["label"],
                    "aggregated_surprise": local_agg,
                    "aggregated_surprise_all": all_agg,
                    "aggregated_surprise_localized": local_agg,
                    "aggregated_surprise_diagnostic_max": local_agg,
                    "aggregated_surprise_diagnostic_mean": pos_diag_mean if item["label"] == "possible" else imp_diag_mean,
                    "aggregated_surprise_old_all_window_max": all_agg,
                    "motion_energy": item["motion_energy"],
                    "condition": item["condition"],
                    "property": item["property"],
                    "source": item["source"],
                    "num_windows": num_windows,
                    "differentiable_windows": differentiable_windows,
                    "active_tokens_mean": active_tokens_mean,
                    "window_surprises": " ".join(f"{x:.6f}" for x in local_scores),
                    "window_surprises_all": " ".join(f"{x:.6f}" for x in all_scores),
                    "window_surprises_localized": " ".join(f"{x:.6f}" for x in local_scores),
                    "diagnostic_window_flags": " ".join("1" if x else "0" for x in diagnostic_flags),
                    "matched_movie_id": matched_id,
                    "differentiable": differentiable,
                    "not_measurable": not_measurable,
                }
                rows_out.append(out)
                append_csv_row(csv_writer, csv_handle, out)
                for example in examples:
                    if example.get("movie_id") == item["movie_id"]:
                        example["score"] = local_agg
                print(
                    f"{set_id} pair={pair_id} {item['movie_id']} {item['label']} "
                    f"localized={local_agg:.6f} all={all_agg:.6f} "
                    f"differentiable={differentiable} active_tokens_mean={active_tokens_mean:.1f}"
                )

        for item in movie_items:
            del item["windows"]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        csv_handle.close()

    # On a resumed run, pair_rows holds only this invocation's pairs, so the
    # in-run report/figures would be partial. Skip them; the authoritative
    # accuracy is computed from the complete CSV once all movies are scored.
    if resumed or not pair_rows:
        scored = len(rows_out)
        print(f"Scored {scored} new movie row(s) this run; CSV now at {csv_path}.")
        print("Run complete-CSV analysis for the powered accuracy (do not trust a partial in-run report).")
        return

    localized = {
        "overall": pair_accuracy(pair_rows, metric="localized"),
        "differentiable": pair_accuracy(pair_rows, metric="localized", subset="differentiable"),
        "undifferentiable": pair_accuracy(pair_rows, metric="localized", subset="undifferentiable"),
    }
    all_token = {
        "overall": pair_accuracy(pair_rows, metric="all"),
        "differentiable": pair_accuracy(pair_rows, metric="all", subset="differentiable"),
        "undifferentiable": pair_accuracy(pair_rows, metric="all", subset="undifferentiable"),
    }
    by_block = breakdown_pair_accuracy(pair_rows, "condition", metric="localized")
    by_property = breakdown_pair_accuracy(pair_rows, "property", metric="localized")

    acc_path = figures_dir / "intphys_probe_accuracy.png"
    compare_path = figures_dir / "intphys_probe_localized_vs_all.png"
    scatter_path = figures_dir / "intphys_probe_surprise_motion.png"
    examples_path = figures_dir / "intphys_probe_examples.png"
    acc_labels = ["overall", "differentiable", "undifferentiable"] + list(by_block)
    acc_values = [localized[k][0] for k in ("overall", "differentiable", "undifferentiable")] + [
        by_block[k][0] for k in by_block
    ]
    acc_counts = [localized[k][1] for k in ("overall", "differentiable", "undifferentiable")] + [
        by_block[k][1] for k in by_block
    ]
    save_accuracy_figure(acc_labels, acc_values, acc_counts, acc_path, title="Localized VoE accuracy")
    save_localized_vs_all_figure(localized, all_token, compare_path)
    save_motion_scatter(rows_out, scatter_path)
    save_examples(examples, examples_path)

    figures = [
        ("localized VoE relative accuracy overall, differentiable, undifferentiable, and per block", str(acc_path)),
        ("localized-vs-all-token accuracy comparison", str(compare_path)),
        ("localized aggregated surprise versus low-level motion energy", str(scatter_path)),
        ("sampled frames from a few scored movies with surprise values", str(examples_path)),
    ]
    write_report(args, source_info, localized, all_token, by_block, by_property, csv_path, figures, len(rows_out))
    print(f"Localized overall relative VoE accuracy: {localized['overall'][0]:.4f} over {localized['overall'][1]} pairs")
    print(
        f"Differentiable pairs: {localized['differentiable'][1]}, "
        f"undifferentiable pairs: {localized['undifferentiable'][1]}"
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
