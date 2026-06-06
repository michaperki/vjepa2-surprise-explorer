# Cross-check: our aggregation vs the paper's shipped surprises

**Question.** We get ~chance on IntPhys; the paper gets ~0.9+. Is the gap in our
*analysis* (turning per-window surprise into pairwise VoE accuracy) or in
surprise *generation* (the model+protocol that produces the surprises)?

**Method (CPU, free).** The paper repo ships `data_intphys.tar.gz` with the raw
per-window surprises for every model it evaluated. We ran *our* aggregation on
*their* numbers. Files saved under `data/paper_intphys_surprises/`; reproduced by
`run_crosscheck.py`. Their format: `losses (movies, contexts, windows)`,
`labels` (0 = impossible, 1 = possible), `context_lengths [2,4,6,8,10]`. Their
"Relative Accuracy (avg)" = per-window-mean surprise, possible/impossible paired
by position within each 4-movie scene, correct when impossible > possible.

## Result 1 — our pipeline is correct (exact reproduction)

Running our aggregation on `vit-l-rope-howto`'s raw surprises reproduces their
published `performance.csv` **exactly**:

```
vit-l-rope-howto O1 Relative Accuracy (avg):
  ours:  [93.33, 95.0, 93.33, 93.33, 91.67]
  paper: [93.33, 95.0, 93.33, 93.33, 91.67]   abs err 0.00  -> PASS
```

So the VoE scoring logic (matched pairs, avg-over-windows, impossible>possible)
is **not** where our null result comes from.

## Result 2 — size is not the blocker; training data is the axis

Best-context IntPhys VoE accuracy (avg-agg, matched pairs), computed by us from
their raw surprises:

| checkpoint | O1 | O2 | O3 | mean | what it is |
| --- | ---: | ---: | ---: | ---: | --- |
| vit-h-rope-howto | 93.3 | 93.3 | 96.7 | **94.4** | ViT-H, HowTo100M |
| **vit-l-rope-howto** | 95.0 | 88.3 | 91.7 | **91.7** | **ViT-L, HowTo100M — same arch as ours** |
| vit-l-rope-k710 | 76.7 | 80.0 | 78.3 | 78.3 | ViT-L, Kinetics-710 |
| videomaev2_g | 55.0 | 65.0 | 58.3 | 59.4 | VideoMAE-v2 giant (pixel pred) |
| vit-l-rope-ssv2 | 61.7 | 38.3 | 40.0 | 46.7 | ViT-L, SSv2 |
| vit-l-rope-random-2 | 51.7 | 51.7 | 50.0 | 51.1 | untrained ViT-L (chance) |

This reproduces the paper's central claim — intuitive-physics VoE *emerges* and
is strongly **training-data-dependent** (HowTo ≫ K710 ≫ SSv2 ≈ random). Crucially,
**the ViT-L architecture our engine builds reaches 91.7%** with the right weights.

## What this changes

Our own context sweep (`run_context_sweep.py`, V-JEPA 2 ViT-L) tops out at
**~0.55–0.62** avg-agg. Combined with the two results above:

- **Not the analysis** — our aggregation reproduces their 0.92 to err=0.00.
- **Not the architecture / size** — ViT-L hits 0.92; ViT-H 0.94.
- **Not the context length** — swept; no context recovers it (`CONTEXT_SWEEP`).

→ **The gap is on the generation side: the checkpoint and/or our surprise
protocol (resolution, window sampling).** Our V-JEPA 2 ViT-L behaves like one of
the *weak* checkpoints (between SSv2 and K710), not like HowTo ViT-L.

**This overturns the earlier headline.** "ViT-L surprise carries no physics
signal beyond a motion confound" is true *of `vitl.pt`*, but it is **not** a fact
about the paradigm, the architecture, or world-models in general — the identical
architecture with V-JEPA 1's HowTo weights gets 92% on the same benchmark with
the same (now-validated) scoring. The motion-confound result describes our
checkpoint's surprise, not the method's ceiling.

## The one experiment that isolates checkpoint vs our-protocol

Run V-JEPA 1's **`vit-l-rope-howto` checkpoint through our engine** (GPU, human
launches):
- reproduces ~0.9 → our engine is faithful and **V-JEPA 2 ViT-L is genuinely the
  weaker model for VoE** (the real, citable finding; note V-JEPA 2 optimized for
  action-conditioned world-modeling, a different objective).
- still ~0.6 → **our generation protocol has a residual mismatch** (resolution
  256-vs-224, our 5-window sampling vs their dense sliding windows), i.e. a bug
  on our side, not the model — fix it and re-evaluate `vitl.pt`.

Either outcome is decisive and replaces the current (over-generalized) negative
result with a precise one. Open per SOUL rule 9 until run.
