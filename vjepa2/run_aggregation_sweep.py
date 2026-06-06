#!/usr/bin/env python3
"""Aggregation sweep for the IntPhys surprise probe — no model, no GPU.

The near-chance VoE accuracy swings with how per-window surprise is pooled into
one number per movie (NULL_CONTROLS.md). This script makes that explicit: it
reads the per-window surprise arrays the probe already stored
(`window_surprises_localized` / `window_surprises_all` + `diagnostic_window_flags`)
and evaluates a battery of aggregations on the *same* 180 matched pairs, each
against the two bars the null controls established:

  bar 1 — beat the motion-only classifier (accuracy 0.583, model-free);
  bar 2 — produce a real |gap| larger than the equivalent-pair noise floor.

This is the honest version of "match the paper's protocol / try aggregation
variants": the published IntPhys VoE protocol is *some* per-video aggregate of a
sliding-window surprise, classified relatively within a pair. There is no single
canonical choice, so we run the plausible ones side by side rather than picking
the flattering one. An aggregation only counts as physics signal if it clears
both bars — otherwise it is reading motion or noise.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# Motion-only baseline accuracy and noise-floor reference come from
# run_null_controls.py (localized metric). Recomputed here so the script is
# self-contained, but these are the numbers to beat.
SURPRISE_COLS = {"localized": "window_surprises_localized", "all_token": "window_surprises_all"}


def parse_floats(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split()], dtype=float)


def aggregations(w: np.ndarray, flags: np.ndarray) -> dict[str, float]:
    """Per-movie scalars from a per-window surprise array `w` and the
    diagnostic-window mask `flags` (1 where possible/impossible differ)."""
    diag = w[flags > 0] if flags.sum() else w
    med = float(np.median(w))
    return {
        "mean": float(w.mean()),
        "max": float(w.max()),
        "median": med,
        "peak_relative": float(w.max() - med),     # peak above the clip's own baseline
        "diag_mean": float(diag.mean()),            # temporal localization: mean over violation windows
        "diag_max": float(diag.max()),              # temporal localization: peak over violation windows
        "last_window": float(w[-1]),
    }


def load(csv_path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(csv_path.open()):
        flags = parse_floats(r["diagnostic_window_flags"]) if r.get("diagnostic_window_flags") else np.ones(1)
        rows.append({
            "set_id": r["set_id"], "pair_id": r["pair_id"], "label": r["label"],
            "block": r["condition"], "motion": float(r["motion_energy"]),
            "windows": {m: parse_floats(r[c]) for m, c in SURPRISE_COLS.items()},
            "flags": flags,
        })
    return rows


def pairwise_accuracy(scalars: dict[tuple, float], rows: list[dict]) -> tuple[float, int, float, np.ndarray]:
    """Given per-movie scalars keyed (set_id, pair_id, label), pair them and
    return accuracy, n, binomial p, and the array of gaps."""
    by: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r["set_id"], r["pair_id"])
        sc = scalars.get((r["set_id"], r["pair_id"], r["label"]))
        if sc is not None:
            by[key][r["label"]] = sc
    gaps = np.array([d["impossible"] - d["possible"] for d in by.values()
                     if "possible" in d and "impossible" in d])
    k = int((gaps > 0).sum())
    n = len(gaps)
    p = float(stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    return k / n, n, p, gaps


def noise_floor(scalars: dict[tuple, float], rows: list[dict]) -> float:
    """Median |gap| between physically-equivalent movies (two possibles / two
    impossibles in a scene) under this aggregation."""
    by_scene: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"possible": [], "impossible": []})
    for r in rows:
        sc = scalars.get((r["set_id"], r["pair_id"], r["label"]))
        if sc is not None:
            by_scene[r["set_id"]][r["label"]].append(sc)
    eq = []
    for d in by_scene.values():
        for lab in ("possible", "impossible"):
            v = d[lab]
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    eq.append(abs(v[i] - v[j]))
    return float(np.median(eq)) if eq else float("nan")


def motion_baseline(rows: list[dict]) -> tuple[float, int]:
    by: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in rows:
        by[(r["set_id"], r["pair_id"])][r["label"]] = r["motion"]
    diff = np.array([d["impossible"] - d["possible"] for d in by.values()
                     if "possible" in d and "impossible" in d])
    return float((diff > 0).mean()), len(diff)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="outputs/intphys_probe_full.csv")
    ap.add_argument("--out", default="AGGREGATION_SWEEP.md")
    args = ap.parse_args()

    rows = load(Path(args.csv))
    mot_acc, mot_n = motion_baseline(rows)

    agg_names = list(aggregations(np.ones(2), np.ones(2)).keys())
    results = []
    for metric in SURPRISE_COLS:
        for agg in agg_names:
            scalars = {(r["set_id"], r["pair_id"], r["label"]): aggregations(r["windows"][metric], r["flags"])[agg]
                       for r in rows}
            acc, n, p, gaps = pairwise_accuracy(scalars, rows)
            floor = noise_floor(scalars, rows)
            real = float(np.median(np.abs(gaps)))
            results.append({
                "metric": metric, "agg": agg, "acc": acc, "n": n, "p": p,
                "real_abs_gap": real, "floor": floor,
                "floor_ratio": real / floor if floor else float("nan"),
                "beats_motion": acc > mot_acc,
            })

    results.sort(key=lambda d: -d["acc"])
    L = ["# Aggregation sweep — can any pooling extract physics signal?\n"]
    L.append(f"CPU-only, from `{args.csv}`. 180 matched pairs, per-movie scalar via each "
             "aggregation of the stored per-window surprise. Bars from NULL_CONTROLS.md:\n")
    L.append(f"- **Motion-only baseline accuracy: {mot_acc:.4f}** (n={mot_n}, model-free). An "
             "aggregation must beat this to be more than a motion detector.")
    L.append("- **Noise floor**: real |gap| must exceed the equivalent-pair |gap| "
             "(`floor ratio` > 1) to be signal rather than noise.\n")
    L.append("| metric | aggregation | accuracy | p vs 0.5 | real \\|gap\\| | floor | floor ratio | beats motion? |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |")
    for d in results:
        L.append(f"| {d['metric']} | {d['agg']} | {d['acc']:.4f} | {d['p']:.4f} | "
                 f"{d['real_abs_gap']:.5f} | {d['floor']:.5f} | {d['floor_ratio']:.2f}× | "
                 f"{'yes' if d['beats_motion'] else 'no'} |")

    best = max(results, key=lambda d: d["acc"])
    any_clears = [d for d in results if d["beats_motion"] and d["floor_ratio"] > 1.0]
    L.append("\n## Bottom line\n")
    L.append(f"- Best aggregation: **{best['metric']}/{best['agg']}** at {best['acc']:.4f} "
             f"(motion baseline {mot_acc:.4f}).")
    if any_clears:
        names = ", ".join(f"{d['metric']}/{d['agg']} ({d['acc']:.3f})" for d in any_clears)
        L.append(f"- Aggregations clearing **both** bars (beat motion AND |gap| > noise floor): {names}. "
                 "These are the only candidates worth a model-based follow-up.")
    else:
        L.append("- **No aggregation clears both bars.** Nothing here beats the motion baseline "
                 "while also producing a |gap| above the equivalent-pair noise floor. The choice "
                 "of pooling moves the accuracy around chance but never extracts a physics signal "
                 "that the model contributes over motion. This is a CPU-confirmed ceiling: the "
                 "remaining hope for signal is a *different surprise*, not a different aggregation "
                 "— i.e. finer spatial+temporal localization at the violation (the GPU "
                 "violation-window scorer) or reducing distribution shift (the fps sweep).")
    Path(args.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
