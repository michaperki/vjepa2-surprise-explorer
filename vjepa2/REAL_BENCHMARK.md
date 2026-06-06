# Real benchmark surprise probe

## Data slice

Source: original IntPhys dev local folder `dev`. Official dev size is 3 GB archive; 30 scenes per block O1/O2/O3; this run uses blocks `O1,O2,O3` and up to `15` matched scenes.
Movies scored: `60`

Original JEPA intuitive-physics repo survey: its bundled `data_intphys.tar.gz` is precomputed model surprises/performance files, not raw videos. The official raw IntPhys dev archive is 3 GB, with 30 scenes per block and folders `dev/O1|O2|O3/<scene>/<1..4>/scene/*.png` plus `status.json` labels. This runner can consume that local structure, but does not auto-download the 3 GB archive.

## Protocol

Each movie is sampled into 16-frame windows with `frame_step=2` and `stride=2` over sampled frames. Each window is scored by `SurpriseEngine`: first 8 frames are context and last 8 are target, using the existing V-JEPA 2 ViT-L predictor, target layer norm, and mean absolute embedding error. Movie surprise is aggregated by `max` over window scores. This mirrors the JEPA intuitive-physics protocol's sliding-window losses and relative possible-vs-impossible comparison, while using our fixed ViT-L engine.

Localized metric: within each matched possible/impossible pair and aligned target window, the active-token mask is built from target-frame pixel differences using the same 16x16-per-slot token-grid localization as `run_physics_probe_v2.py`. Empty active masks are marked `undifferentiable` and use all-token values as the localized fallback only for CSV completeness; accuracy is reported separately for differentiable and undifferentiable subsets.

Model: this is the ViT-L checkpoint, run deliberately (it fits the hardware). The question we ask is whether *this* model's surprise separates possible from impossible, not whether a larger one would. We make no claim that ViT-Huge would close the gap — that is an untested assumption, not a result; testing it is a separate, open experiment.

## Accuracy

Localized overall relative VoE accuracy: `0.4667` over `30` matched pairs.
Differentiable pairs: `30`; undifferentiable pairs: `0`.
Localized differentiable accuracy: `0.4667` over `30` pairs.
Localized undifferentiable accuracy: `nan` over `0` pairs.

All-token overall accuracy: `0.3667` over `30` matched pairs.
All-token differentiable accuracy: `0.3667` over `30` pairs.
All-token undifferentiable accuracy: `nan` over `0` pairs.

Localized by block/condition:
- `O1`: `0.5000` over `20` sets
- `O2`: `0.4000` over `10` sets

Localized by physical property:
- `O1`: `0.5000` over `20` sets
- `O2`: `0.4000` over `10` sets

CSV: `outputs/intphys_probe_movies.csv`

Figures:
- `figures/intphys_probe_accuracy.png`: localized VoE relative accuracy overall, differentiable, undifferentiable, and per block
- `figures/intphys_probe_localized_vs_all.png`: localized-vs-all-token accuracy comparison
- `figures/intphys_probe_surprise_motion.png`: localized aggregated surprise versus low-level motion energy
- `figures/intphys_probe_examples.png`: sampled frames from a few scored movies with surprise values
