# Controlled physics surprise probe v2

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=. python3 run_physics_probe_v2.py --seeds 10
```

Device: `cuda`
Precision: `bf16`

Design: each trial renders possible, impossible, and naive-linear target continuations from the same 8-frame context. All three variants share byte-identical context frames and are scored with `SurpriseEngine.compute_surprises(...)`, so the V-JEPA prediction from the shared context is reused. Surprise math is unchanged: target-encoder embeddings are layer-normalized and compared to predictor outputs with mean absolute embedding error, matching V-JEPA pretraining with `loss_exp = 1.0`.

Localized metric: for every trial, the active-region token mask is computed from pixel differences among the three target continuations. A target token is active if either frame in its tubelet and its 16x16 spatial patch contains any differing pixel. The same active mask is used for possible, impossible, and linear scores.

Matched gravity rationale: the context moves at near-constant velocity and does not reveal acceleration direction. At the target divergence, possible and impossible continuations both receive equal-magnitude acceleration; possible accelerates downward and impossible accelerates upward. This keeps extrapolation difficulty matched while flipping physical plausibility.

Caveat: these synthetic clips are out-of-distribution for V-JEPA 2. Continuity and solidity still carry some abruptness/motion-distribution confound; the naive-linear baseline is included to expose whether lower surprise reflects physical expectation or simply straight-line extrapolation. Absolute values should not be compared to natural-video runs.

## Localized headline results

| Type | Possible | Impossible | Linear | Impossible > possible | Possible < linear | Impossible-possible gap | Linear-possible gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gravity | 0.613869 | 0.598534 | 0.610825 | 0.00% | 30.00% | -0.015335 | -0.003043 |
| continuity | 0.592761 | 0.590625 | 0.592761 | 40.00% | 0.00% | -0.002136 | 0.000000 |
| solidity | 0.622063 | 0.615079 | 0.615079 | 10.00% | 10.00% | -0.006984 | -0.006984 |

## All-token dilution check

| Type | Possible all-token | Impossible all-token | Linear all-token |
| --- | ---: | ---: | ---: |
| gravity | 0.568074 | 0.565284 | 0.564329 |
| continuity | 0.565806 | 0.563874 | 0.565806 |
| solidity | 0.576627 | 0.569932 | 0.569932 |

## Figures

- `figures/physics_v2_gravity_localized_bars.png`: gravity localized grouped bar chart for possible/impossible/linear
- `figures/physics_v2_gravity_impossible_heatmap_masked.png`: gravity impossible heatmap with active-region mask outline
- `figures/physics_v2_gravity_target_examples.png`: gravity target-frame examples for possible/impossible/linear
- `figures/physics_v2_gravity_localized_vs_all.png`: gravity localized-vs-all-token dilution chart
- `figures/physics_v2_continuity_localized_bars.png`: continuity localized grouped bar chart for possible/impossible/linear
- `figures/physics_v2_continuity_impossible_heatmap_masked.png`: continuity impossible heatmap with active-region mask outline
- `figures/physics_v2_continuity_target_examples.png`: continuity target-frame examples for possible/impossible/linear
- `figures/physics_v2_continuity_localized_vs_all.png`: continuity localized-vs-all-token dilution chart
- `figures/physics_v2_solidity_localized_bars.png`: solidity localized grouped bar chart for possible/impossible/linear
- `figures/physics_v2_solidity_impossible_heatmap_masked.png`: solidity impossible heatmap with active-region mask outline
- `figures/physics_v2_solidity_target_examples.png`: solidity target-frame examples for possible/impossible/linear
- `figures/physics_v2_solidity_localized_vs_all.png`: solidity localized-vs-all-token dilution chart
