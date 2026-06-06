# Aggregation sweep — can any pooling extract physics signal?

CPU-only, from `outputs/intphys_probe_full.csv`. 180 matched pairs, per-movie scalar via each aggregation of the stored per-window surprise. Bars from NULL_CONTROLS.md:

- **Motion-only baseline accuracy: 0.5833** (n=180, model-free). An aggregation must beat this to be more than a motion detector.
- **Noise floor**: real |gap| must exceed the equivalent-pair |gap| (`floor ratio` > 1) to be signal rather than noise.

| metric | aggregation | accuracy | p vs 0.5 | real \|gap\| | floor | floor ratio | beats motion? |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| localized | diag_max | 0.5833 | 0.0304 | 0.00231 | 0.00235 | 0.98× | no |
| localized | peak_relative | 0.5722 | 0.0621 | 0.00125 | 0.00191 | 0.66× | no |
| localized | diag_mean | 0.5722 | 0.0621 | 0.00223 | 0.00200 | 1.11× | no |
| localized | mean | 0.5667 | 0.0862 | 0.00049 | 0.00084 | 0.58× | no |
| all_token | mean | 0.5667 | 0.0862 | 0.00041 | 0.00087 | 0.47× | no |
| all_token | diag_mean | 0.5611 | 0.1173 | 0.00150 | 0.00170 | 0.88× | no |
| all_token | diag_max | 0.5500 | 0.2050 | 0.00203 | 0.00207 | 0.98× | no |
| all_token | peak_relative | 0.4944 | 0.9406 | 0.00111 | 0.00198 | 0.56× | no |
| localized | max | 0.4167 | 0.0304 | 0.00054 | 0.00145 | 0.37× | no |
| all_token | max | 0.3944 | 0.0057 | 0.00050 | 0.00145 | 0.34× | no |
| localized | median | 0.3667 | 0.0004 | 0.00033 | 0.00116 | 0.29× | no |
| all_token | median | 0.3444 | 0.0000 | 0.00021 | 0.00116 | 0.18× | no |
| localized | last_window | 0.1444 | 0.0000 | 0.00000 | 0.00135 | 0.00× | no |
| all_token | last_window | 0.1444 | 0.0000 | 0.00000 | 0.00135 | 0.00× | no |

## Bottom line

- Best aggregation: **localized/diag_max** at 0.5833 (motion baseline 0.5833).
- **No aggregation clears both bars.** Nothing here beats the motion baseline while also producing a |gap| above the equivalent-pair noise floor. The choice of pooling moves the accuracy around chance but never extracts a physics signal that the model contributes over motion. This is a CPU-confirmed ceiling: the remaining hope for signal is a *different surprise*, not a different aggregation — i.e. finer spatial+temporal localization at the violation (the GPU violation-window scorer) or reducing distribution shift (the fps sweep).
