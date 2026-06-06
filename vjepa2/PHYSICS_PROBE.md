# Controlled physics surprise probe

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=. python3 run_physics_probe.py --seeds 10
```

Device: `cuda`
Precision: `bf16`

Design: each trial renders a possible/impossible synthetic minimal pair. The first 8 frames are byte-identical and are used as context. The last 8 frames diverge into a physically possible or impossible target. The V-JEPA 2 predictor receives context token slots 0-3 and predicts target token slots 4-7; surprise is the same layer-normalized target-embedding mean absolute error used by the V-JEPA pretraining loss with `loss_exp = 1.0`.

Caveat: these synthetic clips are out-of-distribution for V-JEPA 2. Only within-pair possible-vs-impossible comparisons are meaningful; absolute values should not be compared to natural-video runs.

## Results

| Type | Possible mean | Impossible mean | Gap | Impossible > possible |
| --- | ---: | ---: | ---: | ---: |
| gravity | 0.569794 | 0.567739 | -0.002054 | 30.00% |
| continuity | 0.565422 | 0.564765 | -0.000657 | 50.00% |
| solidity | 0.576659 | 0.570500 | -0.006159 | 0.00% |

## Figures

- `figures/physics_gravity_bars.png`: gravity possible-vs-impossible mean surprise bar chart with error bars
- `figures/physics_gravity_impossible_heatmap.png`: gravity impossible example target-token heatmap
- `figures/physics_gravity_example_pair.png`: gravity possible/impossible target-window example frames
- `figures/physics_continuity_bars.png`: continuity possible-vs-impossible mean surprise bar chart with error bars
- `figures/physics_continuity_impossible_heatmap.png`: continuity impossible example target-token heatmap
- `figures/physics_continuity_example_pair.png`: continuity possible/impossible target-window example frames
- `figures/physics_solidity_bars.png`: solidity possible-vs-impossible mean surprise bar chart with error bars
- `figures/physics_solidity_impossible_heatmap.png`: solidity impossible example target-token heatmap
- `figures/physics_solidity_example_pair.png`: solidity possible/impossible target-window example frames
- `figures/physics_overall_summary.png`: overall per-type possible-vs-impossible summary
