# Reconciliation: our ~0.50 vs the published 0.98 on IntPhys

The V-JEPA team published **0.98 zero-shot VoE accuracy on IntPhys** with the
exact surprise paradigm this repo uses (Garrido, Ballas, LeCun 2025,
[arXiv:2502.11831](https://arxiv.org/abs/2502.11831);
[code](https://github.com/facebookresearch/jepa-intuitive-physics)). Our probe
sits at chance (0.39–0.58 depending on pooling; NULL_CONTROLS.md). A 48-point
gap on the same benchmark, from the same group, is **not a clean negative result
— it is an unreconciled reproduction gap**, and a skeptic asks about it first
(SOUL rule 7). The defensible claim *today* is the narrow one: *"At V-JEPA 2
ViT-L, surprise on IntPhys is a motion confound; we have not reproduced the
published ViT-H 0.98."* This file maps the gap and the plan to close it.

## Protocol diff (verified from the paper + our code)

| Axis | Published 0.98 | Ours | Match? |
| --- | --- | --- | --- |
| Surprise metric | L1 in rep space vs layer-normed target | mean-abs, `loss_exp=1`, layer-normed target (`surprise_engine.py:218`) | ✅ same |
| Eval masking | context frames fully visible → predict all future-frame tokens (no pretraining-style tube mask at eval) | identical (`_make_masks`, past block → future block) | ✅ same |
| Window length | 16 (also tested 32) | 16 (`FRAMES_PER_CLIP`) | ✅ (one of their settings) |
| Frame rate | skip-2 → 7.5fps (≈ training 5.33fps) | skip-2 in REAL_BENCHMARK; sweepable | ✅ roughly |
| Pairwise aggregation | average over windows | tested (all-token mean → 0.567) | ✅ tested, still chance |
| **Context length** | **per-property-optimal; small C (C=2 → predict 14) drives IntPhys** | **fixed 8/8, NEVER swept** | ❌ **untested axis** |
| Model | **V-JEPA 1, ViT-Huge, 224×224** | **V-JEPA 2, ViT-L, 256×256** | ❌ different version + size |

Two axes diverge. Note `IMG_SIZE=256` is **correct for our checkpoint** (V-JEPA 2
ViT-L's native eval resolution per `configs/eval/vitl/*`), not a bug — it differs
from the paper only because the paper used a different model.

## Ranked candidate causes (cheapest / most-likely-fixable first)

1. **Context length (the lever).** The paper's headline IntPhys numbers come from
   sweeping context size per property and small contexts work well; we have only
   ever run 8/8. The aggregation sweep varied *pooling*, the fps sweep varied
   *stride* — neither varied the *context/target split*. This is the one knob the
   paper explicitly credits for IntPhys that we never touched.
   → **`run_context_sweep.py`** (wired, GPU, human-launched). Sweeps
   C ∈ {2,4,6,8,10,12,14}, reports VoE accuracy (max- and mean-agg) per context
   and per block against the motion baseline.

2. **Model version + size.** V-JEPA 2 ViT-L @256 (ours) vs V-JEPA 1 ViT-H @224
   (theirs). A 48-point gap is too large to be size alone, but it is a real
   difference. Per SOUL rule 9 this is **not** an excuse to soften the negative
   result — ViT-H @224 is the literal published config, so running it would be
   *reproduction*, not a hedge. **Constraint:** this box is ~7.6GB RAM and `vitl.pt`
   already needs `mmap` to load (memory: wsl2-ram-checkpoint-oom); ViT-H likely
   will not fit locally. If the context sweep stays at chance, the honest closing
   statement is "context length doesn't recover it at ViT-L; the published config
   is ViT-H, untested here — open question, not closed."

3. **Frame-rate / distribution shift.** Already wired (`run_fps_sweep.py`).
   Secondary to context length but cheap; worth running in the same session.

## Decisive *free* check (no GPU) — do this too

The repo ships `raw_surprises/` (their per-video surprise outputs). Pull the
IntPhys ones and run our pairwise aggregation on them. If we reproduce ~0.98 from
**their** numbers, our analysis/labeling is correct and the entire gap lives in
surprise *generation* (model + context). If we *don't*, the bug is in our
aggregation/labeling and no GPU is needed at all. Either way it isolates the gap.
(Not yet done — needs their `.pth` download.)

## Commands

```bash
# Context sweep — the reconciliation run (human launches, SOUL rule 8):
PYTHONPATH=. python3 run_context_sweep.py --limit 8 --contexts 2,4,8 --dry-run   # smoke (CPU)
PYTHONPATH=. python3 run_context_sweep.py --limit 60 --contexts 2,4,6,8,10,12,14 --weights-dtype bf16
PYTHONPATH=. python3 run_context_sweep.py --all     --contexts 2,4,6,8,10,12,14 --weights-dtype bf16

# Frame-rate sweep (secondary):
PYTHONPATH=. python3 run_fps_sweep.py --limit 60 --steps 1,2,3,4,6 --weights-dtype bf16
```

## Reading the result

- **Any context beats motion baseline (0.583) AND clears the equiv-pair noise
  floor** → the 8/8 split was hiding signal; the negative result was a protocol
  artifact. Re-run NULL_CONTROLS at the winning context before claiming physics.
- **No context clears it** → the strongest negative result this project can make:
  the one knob the paper credits for IntPhys does not rescue V-JEPA 2 ViT-L. The
  remaining gap is the model (ViT-H, untested here) — stated as open, per rule 9.
