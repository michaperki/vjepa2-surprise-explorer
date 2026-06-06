# VIEWER.md

A local research viewer for the V-JEPA 2 physics-surprise work. It is a **lens,
not a runner**: experiments run from the terminal exactly as before; each run
drops a `manifest.json` + `assets/`; the viewer reads that and makes the results
browsable. It embodies SOUL.md — it exists to make the failure boundary visible
(here: that ViT-L surprise sits near chance on IntPhys), not to make results look
finished.

---

## 1. The contract: `manifest.json`

Every run writes one `manifest.json` into its run directory, alongside an
`assets/` folder of MP4s. The current producer is the dense re-score adapter
(`viewer/adapters/intphys_rescore.py`).

```
runs/intphys_rescore/
├── manifest.json
└── assets/
    ├── O1-15_p4/possible.mp4
    ├── O1-15_p4/impossible.mp4
    └── ...
```

```json
{
  "run": {
    "id": "intphys_rescore",
    "commit": "204698b",
    "config_path": "checkpoints/vitl.pt",
    "command": "PYTHONPATH=. python3 -m viewer.adapters.intphys_rescore --all ...",
    "created": "2026-06-05T09:32:00Z",
    "notes": "Dense per-pair re-score ... Population accuracy: 84/180 = 0.467 ..."
  },
  "examples": [
    {
      "id": "O1:15_p4",
      "label": "correct",                  // "correct" if impossible > possible, else "wrong"
      "video_possible":   "assets/O1-15_p4/possible.mp4",
      "video_impossible": "assets/O1-15_p4/impossible.mp4",
      "fps": 12,
      "n_frames": 100,
      "metrics": {
        "surprise_gap": 0.005872,          // impossible mean − possible mean (the VoE signal)
        "possible": 0.560083,
        "impossible": 0.565955,
        "n_windows": 85
      },
      "dense_curve": {                      // surprise per sliding window, aligned to clip time
        "center":     [11.5, 12.5, "..."], // window center in played-frame units
        "possible":   [0.55, 0.55, "..."],
        "impossible": [0.56, 0.56, "..."]
      }
    }
  ]
}
```

**Schema rules**

- Required per example: `id`, `video_possible`, `video_impossible`, `n_frames`,
  `dense_curve`. `label` and `metrics.surprise_gap` drive the Population tab.
- All asset paths are relative to the run directory.
- `surprise_gap > 0` means the impossible clip surprised the model more — the
  physically correct direction. The headline result is that this holds only
  ~half the time (chance).

A small `viewer/manifest.py` helper (`Manifest` with `add_example(...)` /
`write()`) keeps emitting this to a few lines at the end of any adapter.

---

## 2. The three tabs

**Home.** A narrative, not a dashboard: what the project is, what we've done
(including the negative headline result, stated plainly), and what we're
considering next (generate our own controlled test videos; null/sanity-control
baselines; aggregation-metric variants). A live chip line shows the current
run's pair count and VoE accuracy against chance (0.500).

**Examples.** The keeper view. Pick a pair → its possible and impossible clips
play in lockstep, with a playhead riding both dense surprise curves so you can
see *where* in the clip surprise rises (or fails to). Sort the list by any
metric (default: `surprise_gap`). This is the identifiability-figure idea made
interactive, specialized to the possible-vs-impossible comparison.

**Population.** The headline VoE result made browsable: overall accuracy vs.
chance, the `surprise_gap` distribution (a histogram centered on zero — the
visual statement that the model barely separates the two), a per-block
breakdown, and a sortable pair list (default: most-decisive `|gap|` first).
Click any row to jump into that pair's player. This is the one aggregate that
earns its place — it *is* the negative result.

---

## 3. Architecture

```
viewer/
├── serve.py              # stdlib http.server: static + asset (HTTP range) serving + /api
├── manifest.py           # Manifest writer used by adapters
├── adapters/
│   ├── _common.py        # git commit, windowing, frame/MP4 helpers
│   └── intphys_rescore.py# the GPU adapter: dense re-score -> MP4 + curve + manifest
└── static/
    ├── index.html        # three tabs
    ├── app.js            # vanilla JS; native <video> + inline SVG (no build step)
    └── style.css
```

`serve.py` serves `static/`, serves run assets with HTTP range support (so video
scrubbing works), and answers `GET /api/runs` and `GET /api/manifest?run=<key>`.

```bash
# View existing runs (multiple --run flags add a run switcher):
python3 viewer/serve.py --run runs/intphys_rescore
# opens http://127.0.0.1:8000

# Produce a run (GPU; the human launches per SOUL.md rule 8):
PYTHONPATH=. python3 -m viewer.adapters.intphys_rescore \
    --all --out runs/intphys_rescore_all --weights-dtype bf16   # full 180-pair population
PYTHONPATH=. python3 -m viewer.adapters.intphys_rescore \
    --all --limit 4 --out runs/smoke --weights-dtype bf16        # smoke test first
```

`--dry-run` fakes the surprise curve (real MP4s still encoded) so the whole
viewer can be exercised on CPU without the checkpoint.

---

## 4. What experiments must emit (the only workflow change)

At the end of a run, write `manifest.json` + `assets/` via `viewer/manifest.py`:
per pair, the two MP4s of the model-seen frames, the dense surprise curve, the
`surprise_gap`/`label`, and `run.command`/`run.commit`/`run.notes` so the figure
is regenerable (SOUL.md rule 1).
