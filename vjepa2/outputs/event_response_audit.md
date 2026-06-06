# Event-response (delta) scoring audit

180 pairs, from `runs/violation_review_all/manifest.json`. Bars: chance **0.500**, motion-only baseline **0.583** (NULL_CONTROLS.md). All metrics re-aggregate the SAME model surprises; event window + baseline are model-independent (pixel divergence).

| metric | accuracy | beats motion? | within-scene anti-symmetry | median \|gap\| |
| --- | ---: | :---: | ---: | ---: |
| whole-clip mean gap (CURRENT badge) | 0.528 | no | 74% | 0.00042 |
| event-window mean gap | 0.467 | no | 87% | 0.00228 |
| delta: event rise over own baseline | 0.472 | no | 86% | 0.00228 |
| peak rise over own baseline | 0.533 | no | 84% | 0.00247 |
| AUC above own baseline (event) | 0.389 | no | 71% | 0.00503 |
| localized (masked) delta | 0.517 | no | 90% | 0.00797 |
| divergence-weighted gap | 0.517 | no | 77% | 0.00126 |
| ORACLE peak-rise anywhere (ceiling) | 0.406 | no | 43% | 0.00079 |
| (ref) violation-scorer localized gap | 0.533 | no | — | 0.01808 |
| (ref) motion-only baseline | 0.583 | — | — | — |

Anti-symmetry = fraction of two-pair scenes whose gaps have OPPOSITE sign (>~70% means the metric cancels within a scene, so it is pinned near chance by construction — the appearance confound).

## Flips: current whole-clip vs delta-event (34/180)

- wrong -> correct under delta: **12**
- correct -> wrong under delta: **22**
- net change in 'correct' count: **-10** (why accuracy barely moves: flips go both ways)

| example | whole_clip | delta_event | peak_rise | localized_delta | current -> delta |
| --- | ---: | ---: | ---: | ---: | :---: |
| O3:16_p2 | -0.00267 | -0.00737 | -0.00647 | -0.01809 | wrong -> wrong |
| O1:06_p3 | -0.00014 | +0.00174 | +0.00261 | +0.02445 | wrong -> correct |
| O1:17_p4 | -0.00016 | +0.00233 | +0.00009 | +0.00269 | wrong -> correct |
| O1:21_p1 | -0.00008 | +0.00045 | +0.00004 | +0.00412 | wrong -> correct |
| O1:23_p2 | -0.00007 | +0.00071 | +0.00243 | +0.02294 | wrong -> correct |
| O1:28_p3 | -0.00101 | +0.00033 | +0.00030 | +0.01084 | wrong -> correct |
| O2:01_p1 | -0.00007 | +0.00142 | +0.00157 | -0.00127 | wrong -> correct |
