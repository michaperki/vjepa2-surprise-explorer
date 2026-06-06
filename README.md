# V-JEPA 2 Surprise Explorer

An interactive explorer for a single question: when a video shows something
**physically impossible** — an object passing through a wall, or vanishing behind
one and never coming back — does V-JEPA 2 notice? We read out the model's own
**"surprise"** (how wrong its internal next-frame prediction turns out to be) and
let you *watch* it light up on the actual frames.

**Live site:** _enable GitHub Pages (Settings → Pages → `main` / `docs`), then it
serves from `https://<user>.github.io/vjepa2-surprise-explorer/`._

## The finding, in one line

Boiled down to a single number per clip, "was the impossible clip more surprising
than its possible twin?" is **near chance (~0.50)** across 180 matched IntPhys
pairs — the surprise tracks **how much is moving** in the frame far more than
whether physics was violated. A cross-check (`vjepa2/CROSS_CHECK.md`) shows this is
specific to **this checkpoint (V-JEPA 2 ViT-L)**: the *same* pipeline reproduces
the published ~92% on a different checkpoint. The explorer is what makes the result
legible — you can see surprise following motion instead of impossibility.

## Layout

```
docs/                     the static Pages site (self-contained; no backend)
  data/<run>/manifest.json  slimmed per-pair surprise curves + heatmaps
  data/<run>/assets/        model-cropped MP4s of each clip pair
vjepa2/
  viewer/                 the viewer source + static export (build_static.py)
  surprise_engine.py      prediction-error / "surprise" readout
  run_*.py                the probes, scorers, and audits
  *.md                    method writeups (CROSS_CHECK, NULL_CONTROLS, ...)
  outputs/                analysis records (CSV/JSON/markdown)
  figures/                result plots
```

## Reproducing the data

The published `docs/` site needs **nothing** to run. To regenerate the underlying
data you need Meta's V-JEPA 2 codebase and weights, which are **not** included here:

```bash
git clone https://github.com/facebookresearch/vjepa2.git
# drop this repo's vjepa2/*.py and vjepa2/viewer/ alongside it, add a checkpoint,
# then re-run the scorer and rebuild the static site:
python3 -m viewer.build_static --run runs/<run> --out ../docs
```

## Attribution

The model and its source code are **V-JEPA 2 by Meta AI**
([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)). This
repository contains only the surprise-analysis code and the explorer built on top
of it; it does not redistribute Meta's source tree. Stimuli are from the IntPhys
benchmark.
