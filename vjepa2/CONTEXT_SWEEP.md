SYNTHETIC DRY RUN — surprise is fake; loaders/windowing/crop/context-split are real.

# Context-length sweep — reconciliation with the published 0.98

6 pairs, 5 windows/movie, frame_step=2. Model-free **motion baseline = 0.5000** on this subset — the bar to beat.

Per-video surprise aggregated two ways: `max` over windows (paper's preferred single-clip score) and `mean` (its pairwise score). VoE accuracy = fraction where impossible > possible.

| context frames | predict frames | mean surprise | acc (max-agg) | acc (mean-agg) | beats motion? |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 2 | 14 | 0.5515 | 0.5000 | 0.6667 | yes |
| 8 | 8 | 0.5499 | 0.5000 | 0.5000 | no |

## Per-block accuracy (max-agg)

| context | O1 |
| ---: | ---: |
| 2 | 0.500 |
| 8 | 0.500 |

- **Some context length beats the motion baseline** (contexts [2]). The fixed 8/8 split was hiding signal; re-run the null controls at the winning context before claiming physics — it must also clear the equivalent-pair noise floor.
