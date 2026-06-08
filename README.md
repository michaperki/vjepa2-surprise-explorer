# V-JEPA 2 Surprise Explorer

### ▶ Live demo — **https://michaperki.github.io/vjepa2-surprise-explorer/**

An interactive explorer for a single question: when a video shows something
**physically impossible** — an object passing through a wall, or vanishing behind
one and never coming back — does V-JEPA 2 notice? We read out the model's own
**"surprise"** (how wrong its prediction of what comes next turns out to be, in
representation space rather than pixels) and let you *watch* it light up on the
actual frames.

## The finding, in one line

Boiled down to a single number per clip, "was the impossible clip more surprising
than its possible twin?" is **near chance (~0.50)** across 180 matched IntPhys
pairs — the surprise tracks **how much is moving** in the frame far more than
whether physics was violated. A cross-check (`vjepa2/CROSS_CHECK.md`) shows this is
specific to **this checkpoint (V-JEPA 2 ViT-L)**: the *same* pipeline reproduces
the published ~92% on a different checkpoint. The explorer is what makes the result
legible — you can see surprise following motion instead of impossibility.

## What's in the explorer

- **Home** — the plain-language story and the headline number.
- **Examples** — the keeper view: a matched possible/impossible pair playing in
  lockstep, the model's surprise drawn underneath as a curve with a playhead, a
  per-patch **surprise heatmap** you can toggle onto the frames, and an honest
  pixel-difference band (what's *visually* different, which is **not** necessarily
  where the physics breaks).
- **Population** — the headline result made browsable: accuracy vs. chance, the
  per-pair gap histogram, and the within-scene mirror plot showing how the two
  gaps in each scene cancel.
- **Inventory** — per-case diagnostics and a human-review queue.

## Run it locally

The published site is self-contained — no backend, no build step. Serve the
`docs/` folder and open it:

```bash
cd docs
python3 -m http.server 8000   # then open http://localhost:8000
```

## Layout

```
docs/                       the static Pages site (self-contained)
  index.html app.js style.css
  data/<run>/manifest.json   slimmed per-pair surprise curves + heatmaps
  data/<run>/assets/         model-cropped MP4s of each clip pair
vjepa2/
  surprise_engine.py         the prediction-error / "surprise" readout
  run_*.py                   the probes, scorers, and audits
  viewer/                    viewer source + static export (build_static.py)
  *.md                       method writeups (CROSS_CHECK, NULL_CONTROLS, ...)
  outputs/  figures/         analysis records and result plots
SOUL.md                      why this project exists and the standard it holds to
```

## Reproducing the data

The published `docs/` site needs **nothing** to run. To regenerate the underlying
data you need Meta's V-JEPA 2 codebase and weights, which are **not** redistributed
here:

```bash
git clone https://github.com/facebookresearch/vjepa2.git
# drop this repo's vjepa2/*.py and vjepa2/viewer/ alongside it, add a checkpoint,
# then re-score and rebuild the static site:
PYTHONPATH=. python3 -m viewer.adapters.violation_review --out runs/violation_review_all --weights-dtype bf16
PYTHONPATH=. python3 -m viewer.build_static --run runs/violation_review_all --out ../docs
```

See `vjepa2/VIEWER.md` for the manifest contract and the full adapter/viewer
workflow.

## Attribution

The model and its source code are **V-JEPA 2 by Meta AI**
([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)). This
repository contains only the surprise-analysis code and the explorer built on top
of it; it does not redistribute Meta's source tree. Stimuli are from the
[IntPhys](https://www.intphys.com/) benchmark.
