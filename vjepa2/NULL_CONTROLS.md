# Null / sanity controls for the IntPhys surprise probe

CPU-only, from `outputs/intphys_probe_full.csv` (90 scenes, 180 matched possible/impossible pairs). The matched movies are frame-aligned and near pixel-identical except at the brief violation; these controls ask whether the near-chance VoE accuracy reflects physics or a low-level confound.

## 1. Headline accuracy + binomial test

| metric | accuracy | k/n | 95% CI (Wilson) | p vs 0.5 | mean gap | median \|gap\| |
| --- | ---: | ---: | :--- | ---: | ---: | ---: |
| localized | 0.5833 | 105/180 | [0.510, 0.653] | 0.0304 | +0.00286 | 0.00231 |
| all_token | 0.3944 | 71/180 | [0.326, 0.467] | 0.0057 | +0.00202 | 0.00050 |

Accuracy spans **0.394–0.583** across aggregations — it straddles chance. A robust physics signal would not flip sign of significance with the pooling choice.

## 2. Label-permutation null

| metric | observed | null mean | null 95% band | empirical p |
| --- | ---: | ---: | :--- | ---: |
| localized | 0.5833 | 0.5000 | [0.428, 0.572] | 0.0144 |
| all_token | 0.3944 | 0.3001 | [0.244, 0.356] | 0.0008 |

## 3. Equivalent-pair noise floor

Within a scene, two possibles (or two impossibles) differ by no physics violation. Their |gap| is the noise floor; the real possible-vs-impossible |gap| must clear it to count as signal.

| metric | real median \|gap\| (n) | equiv median \|gap\| (n) | ratio | p(real>equiv) |
| --- | ---: | ---: | ---: | ---: |
| localized | 0.00231 (180) | 0.00235 (180) | 0.98× | 0.3893 |
| all_token | 0.00050 (180) | 0.00145 (180) | 0.34× | 1.0000 |

## 4. Motion confound

| metric | corr(surprise, motion) | corr(gap, motion_diff) | motion-only acc | surprise acc |
| --- | ---: | ---: | ---: | ---: |
| localized | +0.539 | +0.302 | 0.5833 | 0.5833 |
| all_token | +0.855 | +0.097 | 0.5833 | 0.3944 |

If surprise correlates strongly with motion energy and a motion-only classifier reaches the surprise accuracy, the surprise result is largely a motion detector — accuracy gains may be confound amplification, not physics.

## Figure

- `figures/null_controls.png`: surprise-vs-motion scatter, pair-gap-vs-motion-diff scatter, and |gap| real-vs-noise-floor histogram.

## Bottom line

- The accuracy is an **aggregation artifact**: localized 0.583 (p=0.030, above chance) vs all-token 0.394 (p=0.006, *below* chance) on the identical pairs. Significance flips sign with the pooling choice; a real physics signal would not.
- **No signal above the noise floor**: the real possible-vs-impossible |gap| (0.00231) is not larger than the gap between two physically-equivalent clips (0.00235); ratio 0.98×, p(real>equiv)=0.389. The violation moves surprise no more than swapping in another valid clip does.
- **It's motion, not physics**: surprise correlates r=+0.855 with raw motion energy, and a motion-only classifier (call the higher-motion movie impossible) reaches 0.583 — matching the localized surprise accuracy of 0.583. The above-chance result needs no model.

**Consequence for accuracy work.** "Accuracy > 0.5" is the wrong bar — it is reachable by a motion baseline. Any new metric (violation-window localization, protocol-matched aggregation, fps matching) must clear two harder bars to count as physics signal: **(1) beat the motion-only classifier**, and **(2) produce a real |gap| that exceeds the equivalent-pair noise floor**. Both are reported here as reusable baselines.

