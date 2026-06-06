"""Build a curation inventory from a completed viewer run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CSV_FIELDS = [
    "spec",
    "family",
    "scene",
    "pair_id",
    "possible_movie_id",
    "impossible_movie_id",
    "model_label",
    "review_gap",
    "localized_gap",
    "alltoken_gap",
    "motion_diff",
    "motion_direction",
    "motion_agrees_with_localized",
    "scene_pair_sign",
    "violation_center",
    "window_start",
    "n_active_tokens",
    "mask_available",
    "mask_windows",
    "mask_active_mean",
    "mask_active_peak",
    "mask_coverage_mean",
    "mask_coverage_peak",
    "metadata_quality",
    "visual_status",
    "visual_notes",
    "violation_type",
    "object_type",
]


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _sign(value: float | None) -> int:
    if value is None or value == 0:
        return 0
    return 1 if value > 0 else -1


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    return None if value is None else round(float(value), ndigits)


def _pair_csv_rows(csv_path: Path) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    if not csv_path.exists():
        return {}
    out: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out.setdefault((row["set_id"], row["pair_id"]), {})[row["label"]] = row
    return out


def _mask_stats(example: dict[str, Any]) -> dict[str, Any]:
    heatmap = example.get("heatmap") or {}
    masks = heatmap.get("mask") or []
    grid_h = int(heatmap.get("grid_h") or 0)
    grid_w = int(heatmap.get("grid_w") or 0)
    denom = max(1, grid_h * grid_w)
    active = [sum(int(v) for row in mask for v in row) for mask in masks]
    if not active:
        return {
            "mask_available": False,
            "mask_windows": 0,
            "mask_active_mean": None,
            "mask_active_peak": None,
            "mask_coverage_mean": None,
            "mask_coverage_peak": None,
        }
    mean_active = sum(active) / len(active)
    peak_active = max(active)
    return {
        "mask_available": True,
        "mask_windows": len(active),
        "mask_active_mean": round(mean_active, 3),
        "mask_active_peak": int(peak_active),
        "mask_coverage_mean": round(mean_active / denom, 6),
        "mask_coverage_peak": round(peak_active / denom, 6),
    }


def _scene_pair_signs(records: list[dict[str, Any]]) -> dict[str, str]:
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_scene.setdefault(record.get("scene") or record["spec"].rsplit("_p", 1)[0], []).append(record)

    out: dict[str, str] = {}
    for scene_records in by_scene.values():
        if len(scene_records) < 2:
            for record in scene_records:
                out[record["spec"]] = "single"
            continue
        signs = [_sign(_float_or_none(record.get("gap_localized"))) for record in scene_records]
        if all(signs) and len(set(signs)) > 1:
            label = "anti_symmetric"
        elif all(signs) and len(set(signs)) == 1:
            label = "same_sign"
        else:
            label = "zero_or_unknown"
        for record in scene_records:
            out[record["spec"]] = label
    return out


def build_inventory(manifest: dict[str, Any], csv_path: Path, records_path: Path) -> dict[str, Any]:
    csv_rows = _pair_csv_rows(csv_path)
    records = _read_json(records_path, [])
    records_by_spec = {record["spec"]: record for record in records}
    scene_pair_sign = _scene_pair_signs(records)
    examples = manifest.get("examples") or []

    items = []
    for example in examples:
        spec = example["id"]
        set_id, pair_id = spec.rsplit("_p", 1)
        family, _scene_num = set_id.split(":", 1)
        pair_rows = csv_rows.get((set_id, pair_id), {})
        possible = pair_rows.get("possible", {})
        impossible = pair_rows.get("impossible", {})
        record = records_by_spec.get(spec, {})
        review_gap = _float_or_none((example.get("metrics") or {}).get("surprise_gap"))
        localized_gap = _float_or_none(record.get("gap_localized"))
        alltoken_gap = _float_or_none(record.get("gap_alltoken"))
        motion_diff = _float_or_none(record.get("motion_diff"))
        motion_sign = _sign(motion_diff)
        localized_sign = _sign(localized_gap)
        mask = _mask_stats(example)
        metadata_missing = [
            name
            for name, value in {
                "csv_possible": possible,
                "csv_impossible": impossible,
                "scorer_record": record,
                "review_example": example,
                "mask": mask["mask_available"],
            }.items()
            if not value
        ]

        item = {
            "spec": spec,
            "family": family,
            "scene": set_id,
            "pair_id": pair_id,
            "possible_movie_id": possible.get("movie_id"),
            "impossible_movie_id": impossible.get("movie_id"),
            "model_label": example.get("label"),
            "review_gap": _round_or_none(review_gap),
            "localized_gap": _round_or_none(localized_gap),
            "alltoken_gap": _round_or_none(alltoken_gap),
            "motion_diff": _round_or_none(motion_diff),
            "motion_direction": {
                1: "impossible_higher_motion",
                -1: "possible_higher_motion",
                0: "tie_or_unknown",
            }[motion_sign],
            "motion_agrees_with_localized": bool(motion_sign and localized_sign and motion_sign == localized_sign),
            "scene_pair_sign": scene_pair_sign.get(spec, "unknown"),
            "violation_center": _int_or_none(record.get("violation_center")),
            "window_start": _int_or_none(record.get("window_start")),
            "n_active_tokens": _int_or_none(record.get("n_active_tokens")),
            **mask,
            "metadata_quality": "complete" if not metadata_missing else "missing:" + ",".join(metadata_missing),
            "available_masks": ["pixel_diff_token_mask"] if mask["mask_available"] else [],
            "visual_status": "unreviewed",
            "visual_notes": "",
            "violation_type": "unknown",
            "object_type": "unknown",
            "curation": {
                "metadata_quality": "complete" if not metadata_missing else "incomplete",
                "visual_status": "unreviewed",
                "visual_failure": "unknown",
                "motion_confound": "candidate" if motion_sign and localized_sign and motion_sign == localized_sign else "not_aligned",
                "violation_type_source": "not_available_in_local_intphys_metadata",
                "object_type_source": "not_available_in_local_intphys_metadata",
            },
        }
        items.append(item)

    summary = {
        "n_items": len(items),
        "families": {family: sum(1 for item in items if item["family"] == family) for family in sorted({i["family"] for i in items})},
        "metadata_quality": {
            key: sum(1 for item in items if item["metadata_quality"] == key)
            for key in sorted({item["metadata_quality"] for item in items})
        },
        "mask_available": sum(1 for item in items if item["mask_available"]),
        "motion_agrees_with_localized": sum(1 for item in items if item["motion_agrees_with_localized"]),
        "visual_status": {
            key: sum(1 for item in items if item["visual_status"] == key)
            for key in sorted({item["visual_status"] for item in items})
        },
    }
    return {
        "schema_version": 1,
        "source": {
            "run_id": (manifest.get("run") or {}).get("id"),
            "csv": str(csv_path),
            "records": str(records_path),
        },
        "summary": summary,
        "items": items,
    }


def write_inventory(run_dir: Path, csv_path: Path, records_path: Path) -> tuple[Path, Path]:
    manifest = _read_json(run_dir / "manifest.json", None)
    if manifest is None:
        raise FileNotFoundError(f"missing manifest: {run_dir / 'manifest.json'}")
    inventory = build_inventory(manifest, csv_path, records_path)

    json_path = run_dir / "inventory.json"
    json_path.write_text(json.dumps(inventory, indent=2) + "\n")

    csv_path_out = run_dir / "inventory.csv"
    with csv_path_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in inventory["items"]:
            writer.writerow({key: item.get(key) for key in CSV_FIELDS})
    return json_path, csv_path_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run directory containing manifest.json")
    parser.add_argument("--csv", default="outputs/intphys_probe_full.csv")
    parser.add_argument("--records", default="outputs/violation_scorer.json")
    args = parser.parse_args()
    json_path, csv_path = write_inventory(Path(args.run), Path(args.csv), Path(args.records))
    print(f"Wrote {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
