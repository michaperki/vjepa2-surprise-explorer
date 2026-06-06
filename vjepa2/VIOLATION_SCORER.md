# Violation-window scorer results

180 pairs. Surprise scored only in the pixel-divergence-localized space-time window. Bars: chance 0.5, motion-only baseline 0.5833.

| metric | accuracy | k/n | p vs 0.5 | median |gap| |
| --- | ---: | ---: | ---: | ---: |
| violation-localized | 0.5333 | 96/180 | 0.4124 | 0.01808 |
| all-token (violation window) | 0.5389 | 97/180 | 0.3326 | 0.00333 |
| motion-only baseline | 0.5833 | — | — | — |

**Verdict:** violation-localized accuracy 0.5333 does NOT beat the motion baseline 0.5833. Finer localization does not rescue the result; the null result holds even when scoring exactly where the violation is.

**Within-scene anti-symmetry check** (90 scenes with two pairs): the two gaps have opposite sign in 84% of scenes; median |gap_a + gap_b| = 0.00114 vs median |gap| = 0.01808 (ratio 0.06). Near-perfect cancellation: the localized surprise is measuring an appearance difference that is symmetric across the matched quadruplet, not physical possibility — so it cannot exceed chance by construction. This is the motion/appearance confound resurfacing under tight localization.
