"""Latent-surface adapter: watch a world model's latent state evolve through time.

The dense surprise adapter (`intphys_rescore`) collapses each window to *one*
scalar — and that scalar is exactly where this checkpoint's physics signal got
laundered into a motion confound (see CROSS_CHECK.md). This adapter goes back
*up* from the scalar: for every sliding 16-frame window of a possible/impossible
pair it reads the target encoder's full token field and emits a small latent
*surface* on the same timeline as the surprise curve, so the eye can catch
structure the scalar hides:

  * a shared-PCA trajectory  — the two clips projected into one 2D space; you
    watch them share a path and then *split* (or fail to).
  * effective rank (rail 1)  — participation ratio of the token cloud per window;
    *collapse* shows up as this number dropping.
  * latent velocity vs pixel flow (rail 2) — is a moment of change a *motion*
    event or a *latent reorganization* the pixels don't explain?
  * possible-vs-impossible divergence with a shuffled-pair *null band* drawn in
    the same frame — so "they split" is read against its own baseline, not on
    vibes. The null is a single random derangement of which impossible clip is
    paired with which possible clip.

PCA is *only* the 2D picture; divergence, velocity and effective rank are all in
the raw representation space. Per SOUL.md principle 8 the real run loads the
model (human launches it); `--dry-run` fakes the embeddings (real MP4 still
encoded) so the whole surface + viewer can be exercised end-to-end on CPU.

    # CPU smoke test (synthetic latents, real MP4s, full viewer wiring):
    PYTHONPATH=. python3 -m viewer.adapters.latent_surface \
        --pairs O1:15_p4,O2:08_p1 --out runs/latent_surface --dry-run

    # the real surface, human-launched once the GPU is free:
    PYTHONPATH=. python3 -m viewer.adapters.latent_surface \
        --all --out runs/latent_surface_all --weights-dtype bf16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from viewer.adapters._common import git_commit
from surprise_engine import GRID_H, GRID_T, GRID_W
from viewer.adapters.intphys_rescore import (
    FRAMES_PER_CLIP,
    WINDOW_TARGET_CENTER,
    all_complete_specs,
    encode_mp4,
    load_clip,
    pair_movie_map,
    window_starts,
)
from viewer.manifest import Manifest

LABELS = ("possible", "impossible")


def movie_motion_map(csv_path: Path) -> dict[tuple[str, str, str], float]:
    """(set_id, pair_id, label) -> motion_energy, for the motion-confound controls."""
    import csv

    out: dict[tuple[str, str, str], float] = {}
    for r in csv.DictReader(csv_path.open()):
        if r.get("motion_energy"):
            out[(r["set_id"], r["pair_id"], r["label"])] = round(float(r["motion_energy"]), 6)
    return out


# --- pure surface math (no model, unit-tested) ---------------------------------


def participation_ratio(tokens: np.ndarray) -> float:
    """Effective number of dimensions a (tokens, C) cloud occupies: (Σλ)²/Σλ².

    Pure-numpy mirror of SurpriseEngine.latent_state's GPU computation, kept here
    so --dry-run and the tests don't need torch."""
    centered = tokens - tokens.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, compute_uv=False)
    sq = sv * sv
    denom = float((sq * sq).sum())
    if denom <= 0.0:
        return 0.0
    return float(sq.sum() ** 2 / denom)


def shared_pca(pos: np.ndarray, imp: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Fit ONE 2D PCA on the union of both clips' per-window embeddings and project
    each. Returns (pos_xy, imp_xy, variance_ratio[2]); a shared basis is what makes
    "they share a path then split" a fair visual comparison."""
    union = np.concatenate([pos, imp], axis=0)
    mean = union.mean(axis=0)
    centered = union - mean
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    comp = vt[:2]
    total = float((s * s).sum()) or 1.0
    var_ratio = [round(float(s[i] ** 2 / total), 4) if i < len(s) else 0.0 for i in range(2)]
    proj = lambda a: (a - mean) @ comp.T
    return proj(pos), proj(imp), var_ratio


def step_magnitude(seq: np.ndarray) -> list[float]:
    """L2 change since the previous row; first entry repeats the second so the
    rail has one value per window (a between-windows quantity, frame-aligned)."""
    if len(seq) < 2:
        return [0.0] * len(seq)
    d = np.linalg.norm(np.diff(seq, axis=0), axis=1)
    return [round(float(v), 6) for v in np.concatenate([d[:1], d])]


def pixel_flow(frames: np.ndarray, centers: list[float]) -> list[float]:
    """Mean |Δpixel| in [0,1] between consecutive window-center frames — a cheap,
    honest motion proxy (not optical flow), matching the existing pixel-diff band."""
    idx = [int(min(len(frames) - 1, max(0, round(c)))) for c in centers]
    f = frames[idx].astype(np.float32) / 255.0
    return step_magnitude(f.reshape(len(idx), -1))


def divergence(pos: np.ndarray, imp: np.ndarray) -> list[float]:
    """Per-window L2 between the possible and impossible latent state (raw space)."""
    n = min(len(pos), len(imp))
    d = np.linalg.norm(pos[:n] - imp[:n], axis=1)
    return [round(float(v), 6) for v in d]


def localized_divergence(pos_sp: np.ndarray, imp_sp: np.ndarray) -> np.ndarray:
    """Per-patch ‖pos − imp‖ from the two clips' spatial token fields. Inputs are
    (windows, GRID_H*GRID_W, C); returns (windows, GRID_H, GRID_W) — the spatial
    drill-down of the scalar divergence: *where on the frame* the representations
    differ, not just by how much."""
    n = min(len(pos_sp), len(imp_sp))
    d = np.linalg.norm(pos_sp[:n] - imp_sp[:n], axis=2)  # (n, HW)
    return d.reshape(n, GRID_H, GRID_W)


def _set_id(spec: str) -> str:
    """`O1:01_p2` -> `O1:01` — the scene a pair belongs to."""
    return spec.rsplit("_p", 1)[0]


def _band_from_series(series: dict[int, list[float]]):
    """Per-window-index 5/50/95th percentiles of a {index: [distances]} map."""
    if not series:
        return None
    idx = sorted(k for k, col in series.items() if col)
    return {
        "index": idx,
        "lo": [round(float(np.percentile(series[k], 5)), 6) for k in idx],
        "median": [round(float(np.percentile(series[k], 50)), 6) for k in idx],
        "hi": [round(float(np.percentile(series[k], 95)), 6) for k in idx],
    }


def within_scene_null(pooled: dict[tuple[str, str], np.ndarray], specs: list[str]):
    """The *right* null for "does the latent encode the violation": same scene,
    same physics-validity. For each scene (set_id) it gathers ‖pos_i − pos_j‖ and
    ‖imp_i − imp_j‖ between its different pairs — how far apart the latent puts two
    clips that share the scene but contain *no* possible/impossible contrast. The
    matched divergence (purple) read against this says whether the violation moves
    the representation *more* than ordinary same-scene variation does.

    (Contrast the cross-scene null, which is dominated by scene identity and so
    sits far above any matched pair — useless for the physics question.)"""
    from collections import defaultdict

    scenes: dict[str, list[str]] = defaultdict(list)
    for s in specs:
        scenes[_set_id(s)].append(s)
    series: dict[int, list[float]] = defaultdict(list)
    for members in scenes.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                for label in LABELS:
                    pa, pb = pooled.get((members[a], label)), pooled.get((members[b], label))
                    if pa is None or pb is None:
                        continue
                    k = min(len(pa), len(pb))
                    d = np.linalg.norm(pa[:k] - pb[:k], axis=1)
                    for i in range(k):
                        series[i].append(float(d[i]))
    return _band_from_series(series)


def cross_scene_null(pooled: dict[tuple[str, str], np.ndarray], specs: list[str], seed: int = 0):
    """Shuffled-pair baseline: a random derangement pairs each possible clip with a
    *different scene's* impossible clip. Kept only for contrast — it measures scene
    identity (different scenes are trivially far apart), NOT the physics violation,
    so the matched divergence always sits far below it. Do not read it as a split test."""
    p = len(specs)
    if p < 2:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(p)
    for i in range(p):  # turn any fixed points into a derangement
        if perm[i] == i:
            perm[i], perm[(i + 1) % p] = perm[(i + 1) % p], perm[i]
    series: dict[int, list[float]] = {}
    for i in range(p):
        pa, pb = pooled.get((specs[i], "possible")), pooled.get((specs[perm[i]], "impossible"))
        if pa is None or pb is None:
            continue
        k = min(len(pa), len(pb))
        d = np.linalg.norm(pa[:k] - pb[:k], axis=1)
        for j in range(k):
            series.setdefault(j, []).append(float(d[j]))
    return _band_from_series(series)


def compute_nulls(pooled: dict[tuple[str, str], np.ndarray], specs: list[str], seed: int = 0) -> dict:
    """Both bands + the metric/method strings the manifest carries."""
    return {
        "metric": "L2 between per-window pooled target-encoder embeddings (raw space)",
        "null_divergence": within_scene_null(pooled, specs),
        "null_kind": "within-scene same-validity (possible-vs-possible / impossible-vs-impossible)",
        "cross_scene_divergence": cross_scene_null(pooled, specs, seed=seed),
        "cross_scene_note": "scene-identity baseline; matched pairs sit far below it by construction",
    }


# --- pooled-embedding cache: GPU extraction once, null/metric recompute on CPU ---


def save_latents(path: Path, pooled: dict[tuple[str, str], np.ndarray]) -> None:
    """Dump per-(spec,label) pooled embeddings so any null/metric can be recomputed
    on CPU later without re-loading the model (`--recompute-null`)."""
    arrays = {f"{spec}|{label}": arr for (spec, label), arr in pooled.items()}
    np.savez_compressed(path, **arrays)  # type: ignore  # str keys map to arrays (stub quirk)


def load_latents(path: Path) -> dict[tuple[str, str], np.ndarray]:
    data = np.load(path)
    out = {}
    for key in data.files:
        spec, label = key.split("|")
        out[(spec, label)] = data[key]
    return out


def save_features(path: Path, feats: dict[tuple[str, str], dict]) -> None:
    """Cache the richer per-clip features (per-layer, per-slot, predictor outputs,
    mean spatial field) so the probe / anticipation / dense views can be built and
    re-built on CPU without re-running the model. Keyed `spec|label|field`."""
    arrays = {}
    for (spec, label), fields in feats.items():
        for field, arr in fields.items():
            arrays[f"{spec}|{label}|{field}"] = arr.astype(np.float32)
    np.savez_compressed(path, **arrays)  # type: ignore  # str keys map to arrays (stub quirk)


def load_features(path: Path) -> dict[tuple[str, str], dict]:
    data = np.load(path)
    out: dict[tuple[str, str], dict] = {}
    for key in data.files:
        spec, label, field = key.split("|")
        out.setdefault((spec, label), {})[field] = data[key]
    return out


# --- delta-direction analysis: not just "did they split" (magnitude / 2D shadow)
#     but "which way did they move, and is that direction shared across pairs?" ---


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def pair_delta_directions(examples: list[dict], pooled: dict[tuple[str, str], np.ndarray]):
    """Per-pair full-dimensional Δ = mean_window(impossible − possible), unit-normed.
    This is the *direction of travel* the 2D PCA only casts a shadow of."""
    specs, vecs = [], []
    for e in examples:
        s = e["id"]
        p, i = pooled.get((s, "possible")), pooled.get((s, "impossible"))
        if p is None or i is None:
            continue
        n = min(len(p), len(i))
        specs.append(s)
        vecs.append(_unit((i[:n] - p[:n]).mean(axis=0)))
    return specs, (np.stack(vecs) if vecs else np.zeros((0, 1)))


def finalize_delta_analysis(examples: list[dict], pooled: dict[tuple[str, str], np.ndarray]):
    """Compute all five delta-direction views. Mutates each example's `latent` block
    with the per-pair pieces (#4 shadow_frac, #5 delta_loadings) and returns the
    population dict (#1 cosine matrix, #2 delta-PCA, #3 null). CPU-only; runs from
    cached embeddings + the manifest, so it is part of `--recompute-null`."""
    # #4 shadow fraction — ‖Δ projected into the per-pair 2D state-PCA‖ / ‖Δ full‖.
    # Both numerator and denominator are in raw units (orthonormal projection), so
    # this is the honest "how much of the real movement the map is showing" in [0,1].
    for e in examples:
        L = e.get("latent")
        if not L:
            continue
        pca, div = L["pca"], L["divergence"]
        sf = []
        for w in range(len(div)):
            px, ix = pca["possible"][w], pca["impossible"][w]
            d2 = ((ix[0] - px[0]) ** 2 + (ix[1] - px[1]) ** 2) ** 0.5
            sf.append(round(d2 / div[w], 4) if div[w] > 1e-9 else 0.0)
        L["shadow_frac"] = sf

    specs, D = pair_delta_directions(examples, pooled)
    if len(D) < 2:
        return None
    cosine = D @ D.T  # #1/#3: true high-dim cosine between violation directions
    # #2/#5: PCA of the directions themselves (centered) -> 2D scatter + axis loadings
    mean = D.mean(axis=0)
    centered = D - mean
    _, sv, vt = np.linalg.svd(centered, full_matrices=False)
    k = int(min(4, vt.shape[0]))
    comp = vt[:k]
    total = float((sv * sv).sum()) or 1.0
    var = [round(float(sv[j] ** 2 / total), 4) if j < len(sv) else 0.0 for j in range(k)]
    coords = centered @ comp.T  # (n, k)
    idx = {s: i for i, s in enumerate(specs)}
    for e in examples:
        if e["id"] in idx and e.get("latent"):
            e["latent"]["delta_loadings"] = [round(float(v), 4) for v in coords[idx[e["id"]]]]
    # #3 null: |cos| between random unit vectors in D dims ~ N(0, 1/D); 95th pct ≈ 1.96/√D.
    null95 = round(float(1.96 / np.sqrt(D.shape[1])), 4)
    return {
        "specs": specs,
        "cosine_matrix": [[round(float(v), 3) for v in row] for row in cosine],
        "delta_pca": {
            "xy": [[round(float(x), 4), round(float(y), 4)] for x, y in coords[:, :2]],
            "var": var[:2],
        },
        "shared_axes_var": var,
        "null_cos95": null95,
        "reduction": "mean over windows of (impossible − possible), unit-normalized, full dim",
    }


# --- views computed on the cached features (CPU; part of --recompute-null) ------


def _linear_probe(X: np.ndarray, y: np.ndarray, seed: int = 0, n_null: int = 5):
    """Cross-validated linear-probe accuracy + a label-shuffle null. Returns
    (acc, null_acc, folds) or None if a class is too small to cross-validate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    counts = np.bincount(y)
    folds = int(min(5, counts[counts > 0].min()))
    if folds < 2 or len(np.unique(y)) < 2:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    acc = float(cross_val_score(clf, X, y, cv=cv).mean())
    rng = np.random.default_rng(seed)
    nulls = []
    for j in range(n_null):
        yp = rng.permutation(y)
        cvj = StratifiedKFold(folds, shuffle=True, random_state=seed + j + 1)
        nulls.append(float(cross_val_score(clf, X, yp, cv=cvj).mean()))
    return round(acc, 4), round(float(np.mean(nulls)), 4), folds


def _ridge_r2(X: np.ndarray, y: np.ndarray, seed: int = 0):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(y) < 6 or np.std(y) < 1e-9:
        return None
    reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    r2 = cross_val_score(reg, X, y, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2").mean()
    return round(float(r2), 4)


def _residualize(X: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Remove the linear component of motion `m` from every feature column."""
    m = (m - m.mean()) / (m.std() + 1e-9)
    beta = (X * m[:, None]).mean(axis=0)  # cov(feature, m) with m standardized
    return X - np.outer(m, beta)


def compute_probes(examples: list[dict], pooled: dict[tuple[str, str], np.ndarray],
                   feats: dict[tuple[str, str], dict], seed: int = 0):
    """Decodability of world variables from the frozen latent: is impossibility,
    scene-block, or motion *linearly* present? Includes a label-shuffle null, a
    layerwise-emergence sweep, and a motion-partialled control."""
    BLOCK = {"O1": 0, "O2": 1, "O3": 2}
    Xs, valid, block, motion = [], [], [], []
    per_layer = []
    for e in examples:
        for lab in LABELS:
            key = (e["id"], lab)
            if key not in pooled:
                continue
            Xs.append(pooled[key].mean(axis=0))
            valid.append(0 if lab == "possible" else 1)
            block.append(BLOCK.get(e["meta"]["block"], -1))
            motion.append(e["meta"].get(f"motion_{lab}"))
            per_layer.append(feats.get(key, {}).get("per_layer"))
    X = np.stack(Xs)
    valid = np.array(valid)
    factors = []
    vp = _linear_probe(X, valid, seed)
    if vp:
        factors.append({"name": "possible vs impossible", "acc": vp[0], "null": vp[1], "n": len(valid), "folds": vp[2]})
    blk = np.array(block)
    if (blk >= 0).all():
        bp = _linear_probe(X, blk, seed)
        if bp:
            factors.append({"name": "which scene type (O1/O2/O3)", "acc": bp[0], "null": bp[1], "n": len(blk), "folds": bp[2]})
    out: dict[str, object] = {"factors": factors}
    if all(v is not None for v in motion):
        m = np.array(motion, dtype=float)
        r2 = _ridge_r2(X, m, seed)
        if r2 is not None:
            out["motion_r2"] = r2
        # motion-partialled possible/impossible probe — does separability survive?
        vpp = _linear_probe(_residualize(X, m), valid, seed)
        if vpp:
            out["validity_motion_partialled"] = {"acc": vpp[0], "null": vpp[1]}
    # layerwise emergence of the possible/impossible signal
    if all(pl is not None for pl in per_layer):
        PL = np.stack(per_layer)  # (clips, depth, C)
        layers, accs, nulls = [], [], []
        for layer in range(PL.shape[1]):
            lp = _linear_probe(PL[:, layer, :], valid, seed, n_null=2)
            if lp:
                layers.append(layer)
                accs.append(lp[0])
                nulls.append(lp[1])
        if layers:
            out["layerwise"] = {"layers": layers, "acc": accs, "null": nulls, "target": "possible vs impossible"}
    return out


def compute_anticipation(examples: list[dict], feats: dict[tuple[str, str], dict]) -> None:
    """Per clip: the predictor's output (z_pred) vs the actual future (h_target),
    as a shared-PCA trajectory + a per-window pooled prediction-error curve. The
    half of V-JEPA that surprise collapses to one scalar."""
    for e in examples:
        block = {}
        ok = True
        for lab in LABELS:
            f = feats.get((e["id"], lab))
            if not f or "z_pred" not in f:
                ok = False
                break
            zp, ht = f["z_pred"], f["h_target"]
            err = np.linalg.norm(zp - ht, axis=1)              # true surprise (raw gap)
            # The predictor and target encoder are different networks, so their
            # pooled outputs sit a near-constant distance apart every frame. That
            # offset dominates a raw PCA (the two paths look like one shifted copy
            # of the other) yet carries no information about *when* the model is
            # surprised. Remove it so the de-biased paths start together and only
            # pull apart where the prediction genuinely drifts off the future.
            offset = zp.mean(axis=0) - ht.mean(axis=0)
            zp_aligned = zp - offset
            pred_xy, actual_xy, _ = shared_pca(zp_aligned, ht)
            gap = np.linalg.norm(zp_aligned - ht, axis=1)       # de-biased drift
            # Do the predicted and actual paths *move together* over time? Cosine of
            # the two de-meaned (per-window) trajectories — the honest "did it
            # anticipate the direction of change", with the constant offset and each
            # clip's time-average removed so neither inflates it.
            zd, hd = zp - zp.mean(axis=0), ht - ht.mean(axis=0)
            den = float(np.linalg.norm(zd) * np.linalg.norm(hd))
            comove = float((zd * hd).sum() / den) if den > 1e-9 else 0.0
            block[lab] = {
                "pred_xy": [[round(float(x), 4), round(float(y), 4)] for x, y in pred_xy],
                "actual_xy": [[round(float(x), 4), round(float(y), 4)] for x, y in actual_xy],
                "err": [round(float(v), 6) for v in err],
                "gap": [round(float(v), 6) for v in gap],
                "offset": round(float(np.linalg.norm(offset)), 4),
                "comove": round(comove, 4),
            }
        if ok and e.get("latent"):
            e["latent"]["anticipation"] = block


def compute_dense_pca(examples: list[dict], feats: dict[tuple[str, str], dict]) -> None:
    """Per clip: top-3 PCA over the patch grid -> an RGB segmentation image
    (per-clip basis), mirroring the V-JEPA 2.1 dense-feature visualizations."""
    for e in examples:
        block = {}
        ok = True
        for lab in LABELS:
            f = feats.get((e["id"], lab))
            if not f or "spatial_mean" not in f:
                ok = False
                break
            sm = f["spatial_mean"]  # (HW, C)
            centered = sm - sm.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vt[: min(3, vt.shape[0])].T  # (HW, <=3)
            if proj.shape[1] < 3:
                proj = np.pad(proj, ((0, 0), (0, 3 - proj.shape[1])))
            lo, hi = proj.min(axis=0), proj.max(axis=0)
            rgb = (proj - lo) / (hi - lo + 1e-9)
            grid = rgb.reshape(GRID_H, GRID_W, 3)
            block[lab] = [[[round(float(v), 3) for v in px] for px in row] for row in grid]
        if ok and e.get("latent"):
            e["latent"]["dense_pca"] = block


def compute_views(examples: list[dict], pooled: dict[tuple[str, str], np.ndarray],
                  feats: dict[tuple[str, str], dict], seed: int = 0):
    """All feature-derived views in one CPU pass (probes are population-level;
    anticipation + dense are per-clip, folded into each `latent` block)."""
    compute_anticipation(examples, feats)
    compute_dense_pca(examples, feats)
    try:
        return compute_probes(examples, pooled, feats, seed)
    except Exception as exc:  # probing is best-effort; never block a run on it
        print(f"[latent] probe step skipped: {exc}")
        return None


# --- per-window latent extraction ----------------------------------------------


# Feature fields stacked per clip. Per-window arrays are (W, ...); `pr` is a list.
FEATURE_FIELDS = ("pooled", "spatial", "per_slot", "per_layer", "z_pred", "h_target")
SYNTH_DIM = 16   # feature width used by --dry-run (real runs use the model's 1024)
SYNTH_LAYERS = 24


def synthetic_clip_features(starts: list[int], spec: str, label: str) -> dict:
    """Deterministic fakes for --dry-run: a smooth drifting trajectory that, for the
    impossible clip, kinks partway through (split + velocity bump + rank dip + a
    localized patch hotspot), plus fake per-slot / per-layer / predictor outputs so
    every downstream view and cache can be exercised on CPU."""
    rng = np.random.default_rng(hash((spec, label)) & 0xFFFFFFFF)
    k = len(starts)
    hw, d = GRID_H * GRID_W, SYNTH_DIM
    t = np.linspace(0, 1, max(1, k))
    pooled = np.zeros((k, d), dtype=np.float32)
    pooled[:, :3] = np.stack([np.sin(2.5 * t), np.cos(1.7 * t), 0.4 * t], axis=1)
    pooled += rng.normal(0, 0.02, pooled.shape)
    pr = 9.0 + 0.5 * np.sin(3 * t) + rng.normal(0, 0.05, k)
    spatial = rng.normal(0, 0.05, (k, hw, d)).astype(np.float32) + pooled[:, None, :]
    if label == "impossible" and k:
        c = int(k * 0.6)
        pooled[c:, 0] += np.linspace(0, 0.8, k - c)
        pr[c : min(k, c + 3)] -= 1.5
        block = np.zeros((GRID_H, GRID_W), dtype=bool)
        block[9:13, 6:10] = True
        spatial[c:, block.reshape(hw), 0] += 0.9
    per_slot = (pooled[:, None, :] + rng.normal(0, 0.03, (k, GRID_T, d))).astype(np.float32)
    # per-layer: features "sharpen" with depth; impossible diverges only in late layers.
    depth_ramp = np.linspace(0.2, 1.0, SYNTH_LAYERS)[None, :, None]
    per_layer = (pooled[:, None, :] * depth_ramp + rng.normal(0, 0.03, (k, SYNTH_LAYERS, d))).astype(np.float32)
    h_target = (pooled + rng.normal(0, 0.05, pooled.shape)).astype(np.float32)
    z_pred = (pooled + rng.normal(0, 0.12, pooled.shape)).astype(np.float32)  # predictor is noisier
    return {
        "pooled": pooled, "pr": [round(float(v), 4) for v in pr], "spatial": spatial,
        "per_slot": per_slot, "per_layer": per_layer, "z_pred": z_pred, "h_target": h_target,
    }


def clip_features(engine, sampled: np.ndarray, starts: list[int]) -> dict:
    """Real per-window features for one clip via the single-pass `latent_features`."""
    acc: dict[str, list] = {k: [] for k in FEATURE_FIELDS}
    pr = []
    for s in starts:
        f = engine.latent_features(sampled[s : s + FRAMES_PER_CLIP])
        pr.append(round(float(f["pr"]), 4))
        for key in FEATURE_FIELDS:
            acc[key].append(f[key])
    out: dict[str, object] = {key: np.stack(acc[key], axis=0).astype(np.float32) for key in FEATURE_FIELDS}
    out["pr"] = pr
    return out


# --- pair assembly -------------------------------------------------------------


def build_pair(spec: str, movies: dict, engine, out_dir: Path, fps: int, width: int,
               frame_step: int, stride: int, dry_run: bool, motion: dict | None = None):
    set_id, pid = spec.rsplit("_p", 1)
    block, scene = set_id.split(":")
    mm = movies.get((set_id, pid))
    if not mm or "possible" not in mm or "impossible" not in mm:
        raise SystemExit(f"{spec}: no possible/impossible movie mapping in CSV")

    safe = spec.replace(":", "-")
    assets = out_dir / "assets" / safe
    assets.mkdir(parents=True, exist_ok=True)

    pooled = {}
    eff_rank = {}
    flow = {}
    spatial = {}
    feats: dict[tuple[str, str], dict] = {}   # (spec,label) -> features for features.npz
    centers = None
    n_frames = fps_used = 0
    for label in LABELS:
        sampled = load_clip(block, scene, mm[label])[::frame_step]
        starts = window_starts(len(sampled), stride)
        if not starts:
            raise SystemExit(f"{spec}/{label}: clip too short for one window")
        win_centers = [round(s + WINDOW_TARGET_CENTER, 1) for s in starts]
        cf = synthetic_clip_features(starts, spec, label) if dry_run else clip_features(engine, sampled, starts)
        encode_mp4(sampled, assets / f"{label}.mp4", fps, width)
        pooled[label] = cf["pooled"]
        eff_rank[label] = cf["pr"]
        spatial[label] = cf["spatial"]
        flow[label] = pixel_flow(sampled, win_centers)
        # Cache for the free CPU views: per-layer/per-slot summarized to per-clip
        # (mean over windows), predictor outputs + spatial kept per-window/temporal.
        feats[(spec, label)] = {
            "per_layer": cf["per_layer"].mean(axis=0),     # (depth, C)
            "per_slot": cf["per_slot"].mean(axis=0),        # (GRID_T, C)
            "z_pred": cf["z_pred"],                          # (W, C)
            "h_target": cf["h_target"],                      # (W, C)
            "spatial_mean": cf["spatial"].mean(axis=0),      # (HW, C) for dense-PCA
        }
        centers = win_centers  # possible/impossible share sampling -> same centers
        n_frames, fps_used = len(sampled), fps

    pos_xy, imp_xy, var_ratio = shared_pca(pooled["possible"], pooled["impossible"])
    div = divergence(pooled["possible"], pooled["impossible"])
    dmaps = localized_divergence(spatial["possible"], spatial["impossible"])  # (n,H,W)
    # Mean matched divergence. NOTE: this is a vector norm, so it is ~always > 0 —
    # it is NOT a "did they split" test on its own. The meaningful read (whether it
    # exceeds the within-scene null) is filled into `label` in main(), once the
    # population-level null band is known.
    sep = round(float(np.mean(div)), 6) if div else 0.0
    example = {
        "id": spec,
        "label": "",  # set in main(): "above null" / "below null"
        "metrics": {
            "matched_divergence": sep,
            "pca_var2d": round(sum(var_ratio), 4),
            "min_eff_rank_imp": round(float(min(eff_rank["impossible"])), 4),
        },
        "video_possible": str((assets / "possible.mp4").relative_to(out_dir)),
        "video_impossible": str((assets / "impossible.mp4").relative_to(out_dir)),
        "fps": fps_used,
        "n_frames": n_frames,
        "meta": {
            "block": block,
            "motion_possible": (motion or {}).get((set_id, pid, "possible")),
            "motion_impossible": (motion or {}).get((set_id, pid, "impossible")),
        },
        "latent": {
            "center": centers,
            "pca": {
                "possible": [[round(float(x), 4), round(float(y), 4)] for x, y in pos_xy],
                "impossible": [[round(float(x), 4), round(float(y), 4)] for x, y in imp_xy],
            },
            "pca_var": var_ratio,
            "eff_rank": eff_rank,
            "latent_vel": {lab: step_magnitude(pooled[lab]) for lab in LABELS},
            "flow": flow,
            "divergence": div,
            "divergence_map": {
                "grid": [[[round(float(v), 5) for v in row] for row in g] for g in dmaps],
                "vmin": round(float(dmaps.min()), 6),
                "vmax": round(float(dmaps.max()), 6),
                "grid_h": GRID_H,
                "grid_w": GRID_W,
                "overlay_inset": 0.0,
                "center": centers,
            },
        },
    }
    return example, pooled["possible"], pooled["impossible"], feats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", help="Comma list like O1:15_p4,O2:08_p1")
    parser.add_argument("--all", action="store_true", help="Every complete pair in the CSV")
    parser.add_argument("--limit", type=int, default=None, help="With --all, cap pairs (smoke test)")
    parser.add_argument("--csv", default="outputs/intphys_probe_full.csv")
    parser.add_argument("--out", default="runs/latent_surface")
    parser.add_argument("--checkpoint", default="checkpoints/vitl.pt")
    parser.add_argument("--weights-dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2, help="window hop (2 = lighter than dense rescore)")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--null-seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="synthetic latents, real MP4 (CPU test)")
    parser.add_argument("--recompute-null", action="store_true",
                        help="CPU-only: rebuild the null bands in an existing --out run from its "
                             "latents.npz (no model, no re-encoding). Use after changing the null.")
    parser.add_argument("--force", action="store_true",
                        help="Allow writing into an --out that already holds a run (overwrites it). "
                             "Off by default so a small test run can't clobber a finished full run "
                             "(SOUL rule 10) — give test runs their own --out instead.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if args.recompute_null:
        recompute_null(out_dir, seed=args.null_seed)
        return

    movies = pair_movie_map(Path(args.csv))
    motion = movie_motion_map(Path(args.csv))
    if args.all:
        specs = all_complete_specs(movies)
        if args.limit is not None:
            specs = specs[: args.limit]
        print(f"[latent] --all: {len(specs)} complete pair(s) from {args.csv}")
    elif args.pairs:
        specs = [s.strip() for s in args.pairs.split(",") if s.strip()]
    else:
        raise SystemExit("pass --pairs <list> or --all")
    out_dir = Path(args.out)
    # Guard (SOUL rule 10): never let a small test run silently overwrite a finished
    # run. If --out already holds a manifest, require --force and report what's there.
    existing = out_dir / "manifest.json"
    if existing.exists() and not args.force:
        try:
            n_prev = len(json.loads(existing.read_text()).get("examples", []))
        except Exception:
            n_prev = "?"
        raise SystemExit(
            f"{out_dir} already holds a run ({n_prev} pairs). Refusing to overwrite it.\n"
            f"  • point a test run at its own --out (e.g. {args.out}_smoke), or\n"
            f"  • pass --force to overwrite this one on purpose, or\n"
            f"  • iterate analysis/views for free with: --out {args.out} --recompute-null"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = None
    if not args.dry_run:
        import torch

        from surprise_engine import SurpriseEngine

        wd = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.weights_dtype]
        engine = SurpriseEngine(checkpoint_path=args.checkpoint, weights_dtype=wd)
        print(f"Device {engine.device}  Precision {engine.precision}")

    examples = []
    pooled: dict[tuple[str, str], np.ndarray] = {}
    feats: dict[tuple[str, str], dict] = {}
    for spec in specs:
        print(f"[latent] {spec} ...", flush=True)
        ex, pos_pooled, imp_pooled, pair_feats = build_pair(
            spec, movies, engine, out_dir, args.fps, args.width,
            args.frame_step, args.stride, args.dry_run, motion,
        )
        m = ex["metrics"]
        print(f"  matched-div {m['matched_divergence']:.4f}  "
              f"2D-var {m['pca_var2d']:.2f}  min-rank(imp) {m['min_eff_rank_imp']:.2f}")
        examples.append(ex)
        pooled[(spec, "possible")] = pos_pooled
        pooled[(spec, "impossible")] = imp_pooled
        feats.update(pair_feats)

    save_latents(out_dir / "latents.npz", pooled)
    save_features(out_dir / "features.npz", feats)
    nulls = compute_nulls(pooled, specs, seed=args.null_seed)
    n_above = label_against_null(examples, nulls["null_divergence"])
    delta_analysis = finalize_delta_analysis(examples, pooled)
    probes = compute_views(examples, pooled, feats, seed=args.null_seed)

    notes = (
        f"Latent surface for {len(examples)} pair(s): per-window pooled target-encoder "
        f"embedding (stride={args.stride}), shared-PCA trajectory, effective-rank and "
        f"latent-velocity-vs-pixel-flow rails, and matched possible/impossible divergence "
        f"against a WITHIN-SCENE null (same-scene, same-validity pairs). The cross-scene "
        f"derangement band is kept only for contrast (it measures scene identity, not physics). "
        f"Divergence, velocity and rank are raw-space; PCA is the 2D picture only."
    )
    if args.dry_run:
        notes = "SYNTHETIC DRY RUN — latents are fake (MP4 is real). " + notes
    manifest = Manifest(
        run_id=out_dir.name,
        command=(f"PYTHONPATH=. python3 -m viewer.adapters.latent_surface "
                 f"{'--all' if args.all else '--pairs ' + (args.pairs or '')} --out {args.out}"),
        notes=notes,
        config_path=None if args.dry_run else args.checkpoint,
        commit=git_commit(),
    )
    manifest.examples = examples
    manifest.latent_space = {"stride": args.stride, **nulls,
                             "delta_analysis": delta_analysis, "probes": probes}
    path = manifest.write(out_dir)
    print(f"Population: {n_above}/{len(examples)} pairs exceed the within-scene null "
          f"(matched divergence > same-scene baseline)")
    print(f"Wrote {path} (+ latents.npz, features.npz) with {len(examples)} pair(s).")


def label_against_null(examples: list[dict], within_null) -> int:
    """Set each example's `label` to "above null"/"below null" by comparing its mean
    matched divergence to the within-scene null median (the honest per-pair read),
    and stash `vs_null` (ratio) as a sortable metric. Returns the count above."""
    med = None
    if within_null:
        med = float(np.mean(within_null["median"])) or None
    n_above = 0
    for e in examples:
        d = float(np.mean(e["latent"]["divergence"])) if e["latent"]["divergence"] else 0.0
        if med:
            ratio = round(d / med, 4)
            e["metrics"]["vs_null"] = ratio
            above = d > med
            e["label"] = "above null" if above else "below null"
            n_above += int(above)
        else:
            e["label"] = "n/a"
    return n_above


def recompute_null(out_dir: Path, seed: int = 0) -> None:
    """CPU-only: rebuild the null bands of an existing run from its latents.npz and
    rewrite the manifest — no model, no MP4 re-encoding. This is the whole point of
    caching the embeddings: GPU extraction happens once, null/metric iteration is free."""
    import json

    npz = out_dir / "latents.npz"
    man_path = out_dir / "manifest.json"
    if not npz.exists() or not man_path.exists():
        raise SystemExit(f"need both {npz} and {man_path} (run the GPU pass once first)")
    pooled = load_latents(npz)
    feats = load_features(out_dir / "features.npz") if (out_dir / "features.npz").exists() else {}
    manifest = json.loads(man_path.read_text())
    specs = [e["id"] for e in manifest["examples"]]
    nulls = compute_nulls(pooled, specs, seed=seed)
    n_above = label_against_null(manifest["examples"], nulls["null_divergence"])
    delta_analysis = finalize_delta_analysis(manifest["examples"], pooled)
    probes = compute_views(manifest["examples"], pooled, feats, seed=seed) if feats else None
    manifest["latent_space"] = {**manifest.get("latent_space", {}), **nulls,
                                "delta_analysis": delta_analysis, "probes": probes}
    man_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Recomputed nulls + delta + views from {npz.name}: "
          f"{n_above}/{len(specs)} pairs exceed within-scene null. Rewrote {man_path}.")


if __name__ == "__main__":
    main()
