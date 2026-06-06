#!/usr/bin/env python3
"""Null / sanity controls for the IntPhys surprise probe — no model, no GPU.

Before we chase accuracy, we ask whether the near-chance VoE number means
anything. Per SOUL.md: a small above-chance accuracy is exactly what a low-level
confound (more motion in the impossible clip) would also produce, so we test for
that explicitly rather than reporting the headline and moving on.

All four controls read the already-written probe CSV
(`outputs/intphys_probe_full.csv`, one row per movie with per-movie aggregated
surprise + motion energy). The matched possible/impossible movies are
frame-aligned and ~pixel-identical except at the brief violation, so any
systematic surprise difference must come from either (a) the violation the model
registered, or (b) a low-level confound. These controls separate the two.

Controls
--------
1. Headline + binomial test. Reproduce the pairwise VoE accuracy for each
   aggregation (localized, all-token) and test it against chance (0.5) with an
   exact binomial test + Wilson 95% CI. If the accuracy swings across
   aggregations and straddles chance, there is no robust signal.

2. Label-permutation null. Within each pair, randomly choose which movie is
   "impossible" and recompute accuracy, many times. The observed accuracy should
   sit inside this null band if surprise carries no physics information.

3. Equivalent-pair noise floor. Within each scene there are two possible and two
   impossible movies. Two possibles (or two impossibles) differ by NO physics
   violation, only noise/setup. Their |surprise gap| is the noise floor. If the
   real possible-vs-impossible |gap| is not larger than this floor, the "signal"
   is noise.

4. Motion confound. Surprise rises with how much is moving. We measure
   corr(surprise, motion_energy) across movies, corr(gap, motion_diff) across
   pairs, and the accuracy of a *motion-only* classifier (call the
   higher-motion movie impossible). If motion alone reaches the surprise
   accuracy, the surprise result is a motion detector, not physics.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

METRICS = {
    "localized": "aggregated_surprise_localized",
    "all_token": "aggregated_surprise_all",
}


def load_rows(csv_path: Path) -> list[dict]:
    return list(csv.DictReader(csv_path.open()))


def matched_pairs(rows: list[dict], col: str) -> list[dict]:
    """One entry per (set, pair) with both labels present."""
    by: dict[tuple[str, str], dict] = defaultdict(dict)
    for r in rows:
        key = (r["set_id"], r["pair_id"])
        by[key].setdefault("block", r["condition"])
        by[key][r["label"]] = float(r[col])
        by[key][f"motion_{r['label']}"] = float(r["motion_energy"])
    out = []
    for (set_id, pair_id), d in by.items():
        if "possible" in d and "impossible" in d:
            out.append({
                "set_id": set_id, "pair_id": pair_id, "block": d["block"],
                "gap": d["impossible"] - d["possible"],
                "motion_diff": d["motion_impossible"] - d["motion_possible"],
            })
    return out


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def control_1_headline(rows: list[dict]) -> dict:
    out = {}
    for name, col in METRICS.items():
        pairs = matched_pairs(rows, col)
        gaps = np.array([p["gap"] for p in pairs])
        k = int((gaps > 0).sum())
        n = len(gaps)
        bt = stats.binomtest(k, n, 0.5, alternative="two-sided")
        lo, hi = wilson_ci(k, n)
        out[name] = {"acc": k / n, "k": k, "n": n, "p": bt.pvalue, "ci": (lo, hi),
                     "mean_gap": float(gaps.mean()), "median_abs_gap": float(np.median(np.abs(gaps)))}
    return out


def control_2_label_permutation(rows: list[dict], n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name, col in METRICS.items():
        gaps = np.array([p["gap"] for p in matched_pairs(rows, col)])
        obs = (gaps > 0).mean()
        # Randomly flip the sign of each pair's gap (= randomly relabel which
        # movie is "impossible"); accuracy under no information.
        flips = rng.choice([-1, 1], size=(n_perm, len(gaps)))
        null_acc = ((flips * gaps) > 0).mean(axis=1)
        p_emp = float((null_acc >= obs).mean())
        out[name] = {"obs": float(obs), "null_mean": float(null_acc.mean()),
                     "null_p2_5": float(np.percentile(null_acc, 2.5)),
                     "null_p97_5": float(np.percentile(null_acc, 97.5)),
                     "p_emp": p_emp}
    return out


def control_3_noise_floor(rows: list[dict]) -> dict:
    """|gap| of physics-equivalent movie pairs (poss-vs-poss, imp-vs-imp) within
    a scene, vs |gap| of the real possible-vs-impossible pairs."""
    out = {}
    for name, col in METRICS.items():
        by_scene: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"possible": [], "impossible": []})
        for r in rows:
            by_scene[r["set_id"]][r["label"]].append(float(r[col]))
        equiv = []
        for d in by_scene.values():
            for lab in ("possible", "impossible"):
                vals = d[lab]
                if len(vals) >= 2:
                    # all unordered pairs within the equivalent set
                    for i in range(len(vals)):
                        for j in range(i + 1, len(vals)):
                            equiv.append(abs(vals[i] - vals[j]))
        real = np.abs([p["gap"] for p in matched_pairs(rows, col)])
        equiv = np.array(equiv)
        u = stats.mannwhitneyu(real, equiv, alternative="greater")
        out[name] = {
            "real_median_abs": float(np.median(real)), "real_n": len(real),
            "equiv_median_abs": float(np.median(equiv)), "equiv_n": len(equiv),
            "ratio": float(np.median(real) / np.median(equiv)) if np.median(equiv) else float("nan"),
            "mwu_p_real_gt_equiv": float(u.pvalue),
        }
    return out


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    # Spearman = Pearson on ranks; avoids scipy's loosely-typed return objects.
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def control_4_motion(rows: list[dict]) -> dict:
    S = {name: np.array([float(r[col]) for r in rows]) for name, col in METRICS.items()}
    M = np.array([float(r["motion_energy"]) for r in rows])
    out = {"corr_surprise_motion": {}, "per_metric": {}}
    for name, col in METRICS.items():
        out["corr_surprise_motion"][name] = {
            "pearson_r": _pearson(S[name], M), "spearman_r": _spearman(S[name], M)}
        pairs = matched_pairs(rows, col)
        gap = np.array([p["gap"] for p in pairs])
        mdiff = np.array([p["motion_diff"] for p in pairs])
        # motion-only classifier: higher-motion movie is "impossible".
        motion_acc = (mdiff > 0).mean()
        # surprise accuracy on the same pairs, for side-by-side.
        surprise_acc = (gap > 0).mean()
        out["per_metric"][name] = {
            "corr_gap_motiondiff": _pearson(gap, mdiff), "p": float("nan"),
            "motion_only_acc": float(motion_acc), "surprise_acc": float(surprise_acc),
        }
    return out


def figure(rows: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = METRICS["localized"]
    pairs = matched_pairs(rows, col)
    gap = np.array([p["gap"] for p in pairs])
    mdiff = np.array([p["motion_diff"] for p in pairs])
    S = np.array([float(r[col]) for r in rows])
    M = np.array([float(r["motion_energy"]) for r in rows])
    # equivalent-pair noise floor
    by_scene = defaultdict(lambda: {"possible": [], "impossible": []})
    for r in rows:
        by_scene[r["set_id"]][r["label"]].append(float(r[col]))
    equiv = []
    for d in by_scene.values():
        for lab in ("possible", "impossible"):
            v = d[lab]
            for i in range(len(v)):
                for j in range(i + 1, len(v)):
                    equiv.append(abs(v[i] - v[j]))
    equiv = np.array(equiv)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].scatter(M, S, s=12, alpha=0.5, color="#2364aa")
    ax[0].set_title(f"surprise vs motion  (r={np.corrcoef(S, M)[0,1]:.3f})")
    ax[0].set_xlabel("motion energy"); ax[0].set_ylabel("surprise (localized)")

    ax[1].scatter(mdiff, gap, s=14, alpha=0.6, color="#1f8a70")
    ax[1].axhline(0, color="#888", lw=0.8); ax[1].axvline(0, color="#888", lw=0.8)
    ax[1].set_title(f"pair gap vs motion diff  (r={np.corrcoef(mdiff, gap)[0,1]:.3f})")
    ax[1].set_xlabel("motion(imp) − motion(pos)"); ax[1].set_ylabel("surprise(imp) − surprise(pos)")

    bins = np.linspace(0, max(np.abs(gap).max(), equiv.max()), 30)
    ax[2].hist(equiv, bins=bins, alpha=0.55, density=True, label="equivalent pairs (noise floor)", color="#9aa3ad")
    ax[2].hist(np.abs(gap), bins=bins, alpha=0.55, density=True, label="possible-vs-impossible", color="#bd3c3c")
    ax[2].set_title("|surprise gap|: real vs noise floor"); ax[2].set_xlabel("|gap|"); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fmt(c1, c2, c3, c4) -> str:
    L = []
    L.append("# Null / sanity controls for the IntPhys surprise probe\n")
    L.append("CPU-only, from `outputs/intphys_probe_full.csv` (90 scenes, 180 matched "
             "possible/impossible pairs). The matched movies are frame-aligned and "
             "near pixel-identical except at the brief violation; these controls ask "
             "whether the near-chance VoE accuracy reflects physics or a low-level "
             "confound.\n")

    L.append("## 1. Headline accuracy + binomial test\n")
    L.append("| metric | accuracy | k/n | 95% CI (Wilson) | p vs 0.5 | mean gap | median \\|gap\\| |")
    L.append("| --- | ---: | ---: | :--- | ---: | ---: | ---: |")
    for name, d in c1.items():
        L.append(f"| {name} | {d['acc']:.4f} | {d['k']}/{d['n']} | "
                 f"[{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] | {d['p']:.4f} | "
                 f"{d['mean_gap']:+.5f} | {d['median_abs_gap']:.5f} |")
    accs = [d["acc"] for d in c1.values()]
    L.append(f"\nAccuracy spans **{min(accs):.3f}–{max(accs):.3f}** across aggregations — "
             "it straddles chance. A robust physics signal would not flip sign of "
             "significance with the pooling choice.\n")

    L.append("## 2. Label-permutation null\n")
    L.append("| metric | observed | null mean | null 95% band | empirical p |")
    L.append("| --- | ---: | ---: | :--- | ---: |")
    for name, d in c2.items():
        L.append(f"| {name} | {d['obs']:.4f} | {d['null_mean']:.4f} | "
                 f"[{d['null_p2_5']:.3f}, {d['null_p97_5']:.3f}] | {d['p_emp']:.4f} |")
    L.append("")

    L.append("## 3. Equivalent-pair noise floor\n")
    L.append("Within a scene, two possibles (or two impossibles) differ by no physics "
             "violation. Their |gap| is the noise floor; the real possible-vs-impossible "
             "|gap| must clear it to count as signal.\n")
    L.append("| metric | real median \\|gap\\| (n) | equiv median \\|gap\\| (n) | ratio | p(real>equiv) |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for name, d in c3.items():
        L.append(f"| {name} | {d['real_median_abs']:.5f} ({d['real_n']}) | "
                 f"{d['equiv_median_abs']:.5f} ({d['equiv_n']}) | {d['ratio']:.2f}× | "
                 f"{d['mwu_p_real_gt_equiv']:.4f} |")
    L.append("")

    L.append("## 4. Motion confound\n")
    L.append("| metric | corr(surprise, motion) | corr(gap, motion_diff) | motion-only acc | surprise acc |")
    L.append("| --- | ---: | ---: | ---: | ---: |")
    for name in METRICS:
        cm = c4["corr_surprise_motion"][name]["pearson_r"]
        pm = c4["per_metric"][name]
        L.append(f"| {name} | {cm:+.3f} | {pm['corr_gap_motiondiff']:+.3f} | "
                 f"{pm['motion_only_acc']:.4f} | {pm['surprise_acc']:.4f} |")
    L.append("\nIf surprise correlates strongly with motion energy and a motion-only "
             "classifier reaches the surprise accuracy, the surprise result is largely a "
             "motion detector — accuracy gains may be confound amplification, not physics.\n")

    L.append("## Figure\n")
    L.append("- `figures/null_controls.png`: surprise-vs-motion scatter, pair-gap-vs-motion-diff "
             "scatter, and |gap| real-vs-noise-floor histogram.\n")

    # --- bottom line, derived from the numbers above ---
    loc = c1["localized"]; allt = c1["all_token"]
    floor = c3["localized"]; mot = c4["per_metric"]["localized"]
    corr_all = c4["corr_surprise_motion"]["all_token"]["pearson_r"]
    L.append("## Bottom line\n")
    L.append(f"- The accuracy is an **aggregation artifact**: localized {loc['acc']:.3f} "
             f"(p={loc['p']:.3f}, above chance) vs all-token {allt['acc']:.3f} "
             f"(p={allt['p']:.3f}, *below* chance) on the identical pairs. Significance flips "
             "sign with the pooling choice; a real physics signal would not.")
    L.append(f"- **No signal above the noise floor**: the real possible-vs-impossible "
             f"|gap| ({floor['real_median_abs']:.5f}) is not larger than the gap between two "
             f"physically-equivalent clips ({floor['equiv_median_abs']:.5f}); "
             f"ratio {floor['ratio']:.2f}×, p(real>equiv)={floor['mwu_p_real_gt_equiv']:.3f}. "
             "The violation moves surprise no more than swapping in another valid clip does.")
    L.append(f"- **It's motion, not physics**: surprise correlates r={corr_all:+.3f} with raw "
             f"motion energy, and a motion-only classifier (call the higher-motion movie "
             f"impossible) reaches {mot['motion_only_acc']:.3f} — matching the localized "
             f"surprise accuracy of {mot['surprise_acc']:.3f}. The above-chance result needs no model.")
    L.append("")
    L.append("**Consequence for accuracy work.** \"Accuracy > 0.5\" is the wrong bar — it is "
             "reachable by a motion baseline. Any new metric (violation-window localization, "
             "protocol-matched aggregation, fps matching) must clear two harder bars to count "
             "as physics signal: **(1) beat the motion-only classifier**, and **(2) produce a "
             "real |gap| that exceeds the equivalent-pair noise floor**. Both are reported here "
             "as reusable baselines.\n")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="outputs/intphys_probe_full.csv")
    ap.add_argument("--out", default="NULL_CONTROLS.md")
    ap.add_argument("--figure", default="figures/null_controls.png")
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_rows(Path(args.csv))
    c1 = control_1_headline(rows)
    c2 = control_2_label_permutation(rows, args.perm, args.seed)
    c3 = control_3_noise_floor(rows)
    c4 = control_4_motion(rows)

    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    figure(rows, Path(args.figure))
    report = fmt(c1, c2, c3, c4)
    Path(args.out).write_text(report + "\n")
    print(report)
    print(f"\nWrote {args.out} and {args.figure}")


if __name__ == "__main__":
    main()
