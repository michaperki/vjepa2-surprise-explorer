# Surprise probe run

Command:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache PYTHONPATH=. python3 run_surprise_probe.py
```

Device: `cuda`

Context/target split: 16 sampled frames become 8 temporal token slots with tubelet size 2. Context uses slots 0-3, covering sampled frames 0-7. Target uses slots 4-7, covering sampled frames 8-15.

Training-distance match: V-JEPA pretraining uses `mean(abs(predicted - target) ** loss_exp) / loss_exp` with `loss_exp = 1.0`, after layer-normalizing target-encoder embeddings. This probe reports mean absolute embedding error per target token, then averages over target tokens for the scalar.

Scalar surprise:

- Real future: `0.611471`
- Broken future: `0.597176`
- Expected `real < broken`: `did not hold`

Figures:

- `figures/surprise_contact_sheet.png`: contact sheet of sampled frames, labeled context vs. target.
- `figures/surprise_heatmaps.png`: per-target-token surprise reshaped into four 16x16 target-time heatmaps for real and broken futures.
- `figures/surprise_bars.png`: bar chart comparing scalar surprise for real vs. broken future.
