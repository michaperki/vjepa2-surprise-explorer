#!/usr/bin/env python3
"""Event-response (delta) scoring audit — CPU, reads the viewer manifest only.

Audits the question the whole-clip mean cannot answer:

    Did the model become MORE surprised when the impossible event happened,
    relative to its OWN prior surprise level?  (delta / event-response)

vs the current verdict metric:

    Which clip had higher AVERAGE surprise over the whole clip?  (whole-clip mean)

Everything here is derived from `runs/<run>/manifest.json` (the dense per-window
surprise curve, the per-window pixel-divergence curve, and the per-window 16x16
heatmap grids + violation masks), so no GPU/model is needed — we are only
re-aggregating surprises the model already produced.

The event window is located from the MODEL-INDEPENDENT pixel divergence (where
the two clips actually differ), so none of these metrics are circular with
surprise. Each clip's baseline is its own low-divergence ("quiet") windows, i.e.
its surprise level when nothing is diverging.

    PYTHONPATH=. python3 run_event_response_audit.py \
        --run runs/violation_review_all --out outputs/event_response_audit.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MOTION_BASELINE = 0.583  # NULL_CONTROLS.md model-free motion-only accuracy (the bar to beat)


def masked_curve(grids, masks):
    """Per-window mean surprise over the patches that differ (the violation
    region); falls back to the whole frame for windows with an empty mask."""
    out = []
    for g, mk in zip(grids, masks):
        g = np.asarray(g, dtype=float)
        mk = np.asarray(mk, dtype=bool)
        out.append(float(g[mk].mean()) if mk.any() else float(g.mean()))
    return np.array(out)


def quiet_baseline(curve, pd, frac=0.34):
    """A clip's 'own prior surprise level': mean over its lowest-divergence
    windows (where the two clips agree, so surprise reflects ordinary content,
    not the event). Robust whether the event is early, middle, or late."""
    k = max(1, int(round(len(curve) * frac)))
    quiet_idx = np.argsort(pd)[:k]  # the k windows with the least pixel divergence
    return float(curve[quiet_idx].mean())


def event_window(pd, lead=2, lag=2):
    """Window indices around the pixel-divergence peak. Includes a small lead/lag
    so a slightly lagged surprise response is still captured."""
    e = int(np.argmax(pd))
    lo = max(0, e - lead)
    hi = min(len(pd), e + lag + 1)
    return e, slice(lo, hi)


def metrics_for(ex):
    c = ex["dense_curve"]
    pos = np.asarray(c["possible"], dtype=float)
    imp = np.asarray(c["impossible"], dtype=float)
    pd = np.asarray(ex["pixel_diff"]["value"], dtype=float)
    hm = ex.get("heatmap")
    if hm is not None:
        posm = masked_curve(hm["possible"], hm["mask"])
        impm = masked_curve(hm["impossible"], hm["mask"])
    else:
        posm, impm = pos, imp

    e, ew = event_window(pd)
    has_event = float(pd.max()) > 1e-6
    pb, ib = quiet_baseline(pos, pd), quiet_baseline(imp, pd)
    pbm, ibm = quiet_baseline(posm, pd), quiet_baseline(impm, pd)

    def relu(x):
        return np.maximum(0.0, x)

    out = {
        # M1 — current verdict metric
        "whole_clip": float(imp.mean() - pos.mean()),
        # M2 — absolute gap, but only in the event window
        "event_mean": float(imp[ew].mean() - pos[ew].mean()),
        # M3 — DELTA: rise of each clip over its own quiet baseline, at the event
        "delta_event": float((imp[ew].mean() - ib) - (pos[ew].mean() - pb)),
        # M4 — peak rise over own baseline during the event window
        "peak_rise": float((imp[ew].max() - ib) - (pos[ew].max() - pb)),
        # M5 — area under curve above own baseline across the event window
        "auc_event": float(relu(imp[ew] - ib).sum() - relu(pos[ew] - pb).sum()),
        # M6 — same delta, but on the localized (masked) surprise
        "localized_delta": float((impm[ew].mean() - ibm) - (posm[ew].mean() - pbm)),
        # M7 — divergence-weighted absolute gap (weight each window by how much the
        #      clips differ there, so 'moments of change' dominate)
        "change_weighted": float(((imp - pos) * pd).sum() / max(1e-9, pd.sum())),
        # M8 — ORACLE upper bound: most-generous 'did impossible spike more than
        #      possible' — max rise anywhere over own baseline. Circular (uses
        #      surprise to pick the moment); a ceiling, not a fair metric.
        "oracle_peak_rise": float((imp.max() - ib) - (pos.max() - pb)),
        "_has_event": has_event,
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/violation_review_all")
    ap.add_argument("--records", default="outputs/violation_scorer.json",
                    help="for the model-free motion baseline + the existing localized gap")
    ap.add_argument("--out", default="outputs/event_response_audit.md")
    args = ap.parse_args()

    manifest = json.loads((Path(args.run) / "manifest.json").read_text())
    exs = manifest["examples"]
    rows = {e["id"]: metrics_for(e) for e in exs}

    rec = {}
    if Path(args.records).exists():
        rec = {r["spec"]: r for r in json.loads(Path(args.records).read_text())}

    keys = ["whole_clip", "event_mean", "delta_event", "peak_rise", "auc_event",
            "localized_delta", "change_weighted", "oracle_peak_rise"]
    labels = {
        "whole_clip": "whole-clip mean gap (CURRENT badge)",
        "event_mean": "event-window mean gap",
        "delta_event": "delta: event rise over own baseline",
        "peak_rise": "peak rise over own baseline",
        "auc_event": "AUC above own baseline (event)",
        "localized_delta": "localized (masked) delta",
        "change_weighted": "divergence-weighted gap",
        "oracle_peak_rise": "ORACLE peak-rise anywhere (ceiling)",
    }
    ids = list(rows)
    N = len(ids)

    def acc(key):
        return float(np.mean([rows[i][key] > 0 for i in ids]))

    def antisym(key):
        by = defaultdict(list)
        for i in ids:
            by[i.rsplit("_p", 1)[0]].append(rows[i][key])
        two = [v for v in by.values() if len(v) == 2]
        if not two:
            return float("nan")
        return float(np.mean([(a > 0) != (b > 0) for a, b in two]))

    L = []
    L.append("# Event-response (delta) scoring audit\n")
    L.append(f"{N} pairs, from `{args.run}/manifest.json`. "
             f"Bars: chance **0.500**, motion-only baseline **{MOTION_BASELINE:.3f}** "
             "(NULL_CONTROLS.md). All metrics re-aggregate the SAME model surprises; "
             "event window + baseline are model-independent (pixel divergence).\n")
    L.append("| metric | accuracy | beats motion? | within-scene anti-symmetry | median \\|gap\\| |")
    L.append("| --- | ---: | :---: | ---: | ---: |")
    for k in keys:
        a = acc(k)
        med = float(np.median([abs(rows[i][k]) for i in ids]))
        beat = "yes" if a > MOTION_BASELINE else "no"
        L.append(f"| {labels[k]} | {a:.3f} | {beat} | {antisym(k)*100:.0f}% | {med:.5f} |")
    if rec:
        gl = np.array([rec[i]["gap_localized"] for i in ids if i in rec])
        mot = np.array([rec[i]["motion_diff"] for i in ids if i in rec])
        L.append(f"| (ref) violation-scorer localized gap | {(gl>0).mean():.3f} | "
                 f"{'yes' if (gl>0).mean()>MOTION_BASELINE else 'no'} | — | {np.median(np.abs(gl)):.5f} |")
        L.append(f"| (ref) motion-only baseline | {(mot>0).mean():.3f} | — | — | — |")
    L.append("")
    L.append("Anti-symmetry = fraction of two-pair scenes whose gaps have OPPOSITE sign "
             "(>~70% means the metric cancels within a scene, so it is pinned near chance "
             "by construction — the appearance confound).\n")

    # flips vs the current badge
    flips = [i for i in ids if (rows[i]["whole_clip"] > 0) != (rows[i]["delta_event"] > 0)]
    gained = [i for i in flips if rows[i]["delta_event"] > 0]  # wrong->correct under delta
    lost = [i for i in flips if rows[i]["delta_event"] <= 0]   # correct->wrong under delta
    L.append(f"## Flips: current whole-clip vs delta-event ({len(flips)}/{N})\n")
    L.append(f"- wrong -> correct under delta: **{len(gained)}**")
    L.append(f"- correct -> wrong under delta: **{len(lost)}**")
    L.append(f"- net change in 'correct' count: **{len(gained) - len(lost):+d}** "
             "(why accuracy barely moves: flips go both ways)\n")
    show = ["O3:16_p2"] + [i for i in gained if i != "O3:16_p2"][:6]
    L.append("| example | whole_clip | delta_event | peak_rise | localized_delta | current -> delta |")
    L.append("| --- | ---: | ---: | ---: | ---: | :---: |")
    for i in show:
        if i not in rows:
            continue
        r = rows[i]
        verd = ("wrong" if r["whole_clip"] <= 0 else "correct") + " -> " + \
               ("correct" if r["delta_event"] > 0 else "wrong")
        L.append(f"| {i} | {r['whole_clip']:+.5f} | {r['delta_event']:+.5f} | "
                 f"{r['peak_rise']:+.5f} | {r['localized_delta']:+.5f} | {verd} |")

    report = "\n".join(L) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report)
    print(report)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
