# V-JEPA 2 Surprise & Latent-Surface Explorer

### ▶ Live demo (surprise explorer) — **https://michaperki.github.io/vjepa2-surprise-explorer/**

An interactive explorer for one question: when a video shows something
**physically impossible** — an object passing through a wall, or vanishing behind
one and never coming back — what does V-JEPA 2's world model *do*? We read that out
two complementary ways and let you *watch* it on the actual frames:

- **Surprise** — how wrong the model's prediction of the next moment is, in
  representation space rather than pixels (the established result below).
- **Latent surface** — how the representation *itself* moves, splits, or collapses
  through the clip: an in-progress *identifiability* lens (`vjepa2/LATENT_SURFACE.md`).

## The finding, in one line

Boiled down to a single number per clip, "was the impossible clip more surprising
than its possible twin?" is **near chance (~0.50)** across 180 matched IntPhys
pairs — the surprise tracks **how much is moving** in the frame far more than
whether physics was violated. A cross-check (`vjepa2/CROSS_CHECK.md`) shows this is
specific to **this checkpoint (V-JEPA 2 ViT-L)**: the *same* pipeline reproduces
the published ~92% on a different checkpoint. The explorer is what makes the result
legible — you can see surprise following motion instead of impossibility.

## What's in the viewer

Two tabs, **Home** (the story + this run's live stats) and **Explore**. A matched
possible/impossible pair plays in lockstep with the model's reaction drawn
underneath. A surprise run shows the surprise readout; a latent-surface run adds a
**view switcher** (deep-linked as `#examples:<view>`) over its lenses — *Latent
surface*, *Probes* (linear decodability vs a shuffle null, with layerwise emergence
and a motion-partialled control), *Anticipation* (the predictor's output vs the
actual future), and *Dense features* (per-clip PCA patch segmentation). The two
base readouts:

- **Surprise readout** — the surprise curve with a playhead, a per-patch
  **surprise heatmap** you can toggle onto the frames, and an honest
  pixel-difference band (what's *visually* different, **not** necessarily where the
  physics breaks).
- **Latent surface** — a shared-PCA **map** of both clips' representation
  trajectories (watch them share a path, then split), plus rails for **effective
  rank** (collapse), **latent velocity vs pixel flow** (motion or reorganization?),
  and **possible−impossible divergence** against a within-scene null — with a
  **localized divergence heatmap** showing *where on the frame* the representations
  differ. This is exploratory; no latent result is claimed yet (see
  `vjepa2/LATENT_SURFACE.md`).

## Run it locally

The published `docs/` site (the **surprise explorer**) is self-contained — no
backend, no build step. Serve the folder and open it:

```bash
cd docs
python3 -m http.server 8000   # then open http://localhost:8000
```

To browse a local run (surprise **or** latent surface) with the live viewer:

```bash
cd vjepa2
PYTHONPATH=. python3 viewer/serve.py --run runs/<your_run>   # open #examples
```

## Layout

```
docs/                       the static Pages site (the published surprise explorer)
  index.html app.js style.css
  data/<run>/manifest.json   slimmed per-pair curves/surfaces + heatmaps
  data/<run>/assets/         model-cropped MP4s of each clip pair
vjepa2/
  surprise_engine.py         the prediction-error readout + latent_state (pooled + rank + spatial)
  run_*.py                   the probes, scorers, and audits
  viewer/                    viewer source + static export (build_static.py)
    adapters/intphys_rescore.py   the surprise re-score adapter
    adapters/latent_surface.py    the latent-surface adapter (+ --recompute-null)
  LATENT_SURFACE.md          the latent-surface method, schema, and open questions
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
# drop this repo's vjepa2/*.py and vjepa2/viewer/ alongside it, add a checkpoint, then:

# surprise explorer (rebuilds the published static site):
PYTHONPATH=. python3 -m viewer.adapters.violation_review --out runs/violation_review_all --weights-dtype bf16
PYTHONPATH=. python3 -m viewer.build_static --run runs/violation_review_all --out ../docs

# latent surface (use --limit/--pairs for a quick sample; --recompute-null re-derives
# the nulls on CPU from the cached embeddings, no model reload):
PYTHONPATH=. python3 -m viewer.adapters.latent_surface --all --out runs/latent_surface --weights-dtype bf16
```

See `vjepa2/VIEWER.md` for the manifest contract and `vjepa2/LATENT_SURFACE.md` for
the latent-surface workflow.

## Attribution

The model and its source code are **V-JEPA 2 by Meta AI**
([facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)). This
repository contains only the surprise- and latent-analysis code and the explorer
built on top of it; it does not redistribute Meta's source tree. Stimuli are from
the [IntPhys](https://www.intphys.com/) benchmark.
