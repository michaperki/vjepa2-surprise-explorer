# Latent surface: watching the world model's state move, split, and collapse

**Motivation.** Every IntPhys result in this repo so far rides on *one scalar per
window* — surprise — and that scalar is exactly where this checkpoint's physics
signal got laundered into a motion confound (`CROSS_CHECK.md`, `NULL_CONTROLS.md`).
The latent surface goes back *up* from the scalar: for every sliding 16-frame
window of a possible/impossible pair it reads the target encoder's full token
field and renders a small, temporally-legible surface on the *same timeline* as
the surprise curve, so structure the scalar destroys becomes visible.

This is an instrument for finding hypotheses, not a claim. The numbers below keep
it honest; the viewer is where interpretation begins.

## What it computes

Per window `w` (target encoder over the 16-frame window, layer-normed, as in
`SurpriseEngine.compute_surprise`):

- **pooled state** `z_w ∈ ℝ¹⁰²⁴` — mean over all tokens. The trajectory the
  surface projects, one point per window.
- **effective rank** `PR_w = (Σλ)²/Σλ²` over the token covariance — the number of
  dimensions the token cloud actually occupies. *Collapse* (dead latents, or a
  representation that throws away structure during an occlusion) shows up as this
  dropping. Label-free; the canonical identifiability read. (`SurpriseEngine.latent_state`.)

The adapter (`viewer/adapters/latent_surface.py`) then derives, per pair:

- **shared-PCA trajectory** — *one* 2D PCA fit on the union of both clips' `z_w`,
  each projected through it. A shared basis is what makes "they share a path then
  *split*" a fair comparison. PCA is **only the 2D picture**; everything scored is
  in raw 1024-d space.
- **latent velocity vs pixel flow** — `‖z_w − z_{w-1}‖` against mean `|Δpixel|`
  between window-center frames. The discovery target is a *latent reorganization
  the pixels don't explain* (velocity moves, flow doesn't): the representation
  reshaping without a motion event. The two have different units, so the viewer
  self-normalizes each before overlaying them.
- **divergence with a within-scene null band** — `‖z_w^pos − z_w^imp‖` per window,
  drawn against a **within-scene, same-validity baseline**: for each scene we gather
  `‖pos_i − pos_j‖` and `‖imp_i − imp_j‖` between its different pairs — how far apart
  the latent puts two clips that share the scene but contain *no* possible/impossible
  contrast — and report the 5/50/95th percentile per window index. The matched
  divergence read against this band says whether the violation moves the
  representation *more than ordinary same-scene variation does*. The band is drawn
  *in the same frame* as the signal — a split counts only when the purple line
  leaves the grey.

  > **Why not a shuffled cross-scene null** (the first thing we tried, kept only as
  > a contrast): pairing each possible with a *different scene's* impossible measures
  > **scene identity** (different scenes are trivially ~13× farther apart), so the
  > matched line sits pinned far below it and tells you nothing about physics. That
  > was a real design error; the within-scene null is the fix.

### Honesty rails (read before believing a picture)

1. **Interpretability ≠ identifiability.** "This PC tracks the ball" is a picture,
   not a recovered latent. Anchor every claim on a number (PR, divergence vs null).
2. **PCA directions are not the model's variables.** A linear projection of an
   entangled latent need not be a disentangled world factor. "Latent variable"
   here means *operationalized*, not a ball-position neuron.
3. **The within-scene null still isn't a clean violation isolator.** Two different
   *valid* pairs of a scene differ in content, not just in "no violation," so the
   band mixes scene variation with validity. It is a far better reference than the
   cross-scene null, not a proof. A clean null needs same-scenario re-renders, which
   IntPhys-2 here doesn't ship.

## Manifest schema

Each example carries, alongside `video_possible` / `video_impossible` / `n_frames`:

```json
"latent": {
  "center":     [...],                       // window-center frames (clip timeline)
  "pca":        {"possible": [[x,y],...], "impossible": [[x,y],...]},
  "pca_var":    [v0, v1],                     // variance explained by the 2 shared PCs
  "eff_rank":   {"possible": [...], "impossible": [...]},
  "latent_vel": {"possible": [...], "impossible": [...]},
  "flow":       {"possible": [...], "impossible": [...]},
  "divergence": [...],                        // raw-space ‖pos − imp‖ per window (scalar)
  "divergence_map": {                          // per-patch ‖pos − imp‖ — the spatial drill-down
    "grid": [[[...16x16...]], ...],            //   one 16x16 map per window
    "vmin": .., "vmax": .., "grid_h": 16, "grid_w": 16, "overlay_inset": 0.0
  },
  "shadow_frac":    [...],                     // #4 fraction of Δ shown by the 2D map, per window
  "delta_loadings": [l0, l1, l2, l3]           // #5 this pair's coords on the shared Δ-axes
}
```

The `divergence_map` is the spatial drill-down of the scalar divergence: a toggle
in the viewer paints it on both clips, following the playhead, so you can see
*where on the frame* the representations differ — on the violation/object, or
smeared over motion. (It reuses the surprise-heatmap overlay; built from the
temporal-mean token field, so it is not in `latents.npz` and `--recompute-null`
leaves it untouched.)

Run-level `latent_space` carries both bands, indexed by window order:
`null_divergence` (the within-scene band; `null` if no scene has ≥2 pairs) and
`cross_scene_divergence` (the scene-identity contrast). Per-pair metrics:
`matched_divergence` (mean), `vs_null` (matched ÷ within-scene-null median),
`pca_var2d`, `min_eff_rank_imp`; `label` is `"above null"` / `"below null"`.

## Direction, not just magnitude — the Δ-direction analysis

`divergence` is `‖Δ‖` (magnitude) and the PCA map is a 2D *shadow of the states*.
Neither tells you **which way** `Δ = impossible − possible` points, or whether that
direction is *shared* across pairs. The delta analysis adds that, all from the
cached pooled embeddings (so it rides `--recompute-null`, no GPU):

- **#1 cosine matrix** — full-dim `cos(Δ_i, Δ_j)` between every pair's mean Δ
  direction. Block structure = violations of a kind share a direction.
- **#2 PCA-of-deltas** — each pair's Δ as a point in the PCA *of the directions*
  (not the states); clusters = consistent violation axes.
- **#3 pairwise readout** — hover the matrix to read `cos(A,B)` against the
  random-direction null (`null_cos95 ≈ 1.96/√dim`), so "aligned/opposed/unrelated"
  is judged against chance, not by eye.
- **#4 shadow fraction** — per window, `‖Δ projected into the 2D map‖ / ‖Δ full‖`,
  a rail in `latent.shadow_frac` ∈ [0,1]: how much of the real movement the PCA map
  is actually showing (on `vitl.pt` it's often only ~0.1–0.4 — most of Δ is
  off-screen, which is the honest caveat on reading the 2D split).
- **#5 shared-axis loadings** — `latent.delta_loadings`, the pair's signed
  coordinate on each top Δ-PCA axis (`latent_space.delta_analysis.shared_axes_var`).

These live in `latent_space.delta_analysis` (`specs`, `cosine_matrix`,
`delta_pca`, `null_cos95`, `shared_axes_var`); per-pair pieces are folded into each
`latent` block. The smoke set already shows the **within-scene pairs are
near-antiparallel** (cos ≈ −0.9, far past the ±0.06 null) — the latent-space echo
of the surprise mirror/anti-symmetry. Caveats from the honesty rails still apply:
the reduction is mean-over-windows (a brief violation gets diluted; "Δ at peak"
is a natural alternative), and Δ is pooled, so direction is a summary over patches.

## Extract once, iterate every view for free

The expensive part (forward passes) and the cheap part (PCA, nulls, probes, views)
are separated. A single run captures **everything** via one `latent_features` pass
per window, written next to the manifest:

- `latents.npz` — per-`(spec,label)` pooled embeddings (delta/null/PCA).
- `features.npz` — the richer per-clip features for the planned multi-view
  explorer: `per_layer` (24×C, layerwise emergence), `per_slot` (GRID_T×C,
  temporal probes), `z_pred` + `h_target` (per-window predictor output vs actual,
  the **anticipation** view), and `spatial_mean` (HW×C, the **dense-PCA**
  segmentation view).
- each example carries `meta` (`block`, `motion_possible`, `motion_impossible`)
  for decodability **probes** and the **motion-confound control**.

So any null/metric/probe/view change is a CPU recompute with no model and no
re-encoding:

```bash
PYTHONPATH=. python3 -m viewer.adapters.latent_surface \
    --out runs/latent_surface_vitl --recompute-null
```

**Views** (built on the cache, no re-run — a switcher in the Explore tab, deep-linked
as `#examples:<view>`):

- **Latent surface** — the PCA map, rails, Δ-direction analysis.
- **Probes** — linear decodability of {possible/impossible, scene block, motion} from
  the frozen pooled latent (5-fold CV) vs a label-shuffle null, a **layerwise-emergence**
  curve, and a **motion-partialled** control (does separability survive removing motion?).
- **Anticipation** — the predictor's output vs the actual future, per clip, as a
  shared-PCA trajectory + a prediction-error curve (the half surprise collapses to a scalar).
  *Extraction note (vitl.pt, 360 clips):* pooled `z_pred` and pooled `h_target` are the
  same size (‖·‖ ≈ 17.4) but separated by a **fixed ≈ 7.7-norm vector that points the
  same way for every clip** (cosine to the global mean offset 0.93, min 0.83; the clip-
  specific residual is only ≈ 3.0). It is **not** a token mismatch (both gather the
  identical `masks_pred` set) and **not** a DC/mean-across-channels shift (that component
  is 0.01) — it's a constant translation between the pooled predictor manifold and the
  pooled layer-normed target manifold, the non-cancelling part of the per-token residual
  that survives token-pooling. It does **not** touch the headline surprise number, which
  is a per-token, per-channel error taken *before* any pooling; this offset only appears
  in the pooled trajectory we draw here, so the view **subtracts it** (and each clip's
  time-mean) before plotting. What's left is the real signal: predicted and actual paths
  **move together with cosine ≈ 0.54** (180 clips, range −0.12…0.76) — the predictor
  anticipates the *direction* of representational change but with shrunk magnitude, which
  is why the de-biased dashed path looks like a contracted copy of the solid one. That
  `move-together r` is shown on each panel.
- **Dense features** — per-clip top-3 PCA of the patch grid as an RGB segmentation
  (mirrors the V-JEPA 2.1 dense-feature figures).

Out of scope here (need training / an LLM / robot data): action recognition, video QA,
robot planning, quantitative dense-task eval — listed on Home, not faked as views.

## Running it

**CPU smoke test** (synthetic latents, real MP4s — proves the surface + viewer
end to end without a checkpoint). Use two pairs of the *same* scene so the
within-scene null is defined:

```bash
PYTHONPATH=. python3 -m viewer.adapters.latent_surface \
    --pairs O1:01_p2,O1:01_p4 --out runs/latent_surface_dry --dry-run
PYTHONPATH=. python3 viewer/serve.py --run runs/latent_surface_dry   # open #examples
```

**The real surface** — loads the model, so the human launches it (SOUL rule 8),
once the GPU is free:

```bash
# weak checkpoint (this repo's V-JEPA 2 ViT-L):
PYTHONPATH=. python3 -m viewer.adapters.latent_surface \
    --all --out runs/latent_surface_vitl --weights-dtype bf16

PYTHONPATH=. python3 -m viewer.build_static --run runs/latent_surface_vitl --out ../docs
```

## The experiment that turns this from a viewer into a result

`CROSS_CHECK.md` leaves one decisive question open: is `vitl.pt` *genuinely* the
weak VoE model, or is there a protocol mismatch? V-JEPA 1's `vit-l-rope-howto`
(92% on IntPhys) runs through the **same engine**. So run the same surface on both
and compare what the eye catches in one second:

```bash
# strong checkpoint, identical pipeline:
PYTHONPATH=. python3 -m viewer.adapters.latent_surface \
    --all --out runs/latent_surface_howto \
    --checkpoint checkpoints/vit-l-rope-howto.pt --weights-dtype bf16
```

- **Strong splits cleanly past the null band, weak smears inside it** → the latent
  surface *mechanistically explains* the surprise null: the weak checkpoint doesn't
  separate possible from impossible in representation space, so no scalar built on
  it could. That is the citable, SOUL-grade finding.
- **Both smear** → the separation isn't in the pooled trajectory at this stride;
  the next move is per-token (localized) divergence, not pooled.

## First real run — `vitl.pt`, 180 pairs (stride 2)

Observations, not claims (single checkpoint, no comparison yet):

- **Effective rank ≈ 30 of 1024** (median; min ≈ 22, max ≈ 36). The token cloud is
  far from collapsed-to-a-point, but uses only ~3% of the dimensions. Whether that
  is "low" is meaningless without the strong-checkpoint reference.
- **corr(latent velocity, pixel flow) = 0.18.** The *rate of change* of the pooled
  latent is **not** a motion proxy — notable given that *surprise* on this same
  checkpoint correlates r = 0.855 with motion energy. Surprise and latent velocity
  are different quantities; the motion confound that owns one does not own the other.
- The first headline the adapter printed — "180/180 pairs with positive latent
  separation" — was **vacuous** (separation is a norm, always ≥ 0). Replaced with
  the within-scene-null comparison above. Logged here so the mistake isn't repeated.

None of this is a result yet: it is one checkpoint, and the decisive comparison
(`vit-l-rope-howto` through the same surface) has not been run.
