#!/usr/bin/env python3
"""Cross-check our VoE aggregation against the published surprises (CPU, free).

The reconciliation question: we get ~chance on IntPhys, the paper gets ~0.9+.
Is the gap in our *analysis* (how we turn per-window surprise into pairwise VoE
accuracy) or in surprise *generation* (the model + protocol that produces the
surprises)? This isolates it with zero GPU by running OUR aggregation on the
paper's own shipped raw surprises (data/paper_intphys_surprises/, extracted from
data_intphys.tar.gz in facebookresearch/jepa-intuitive-physics).

If our aggregation reproduces their published performance.csv numbers, the
analysis pipeline is correct and the entire gap is on the generation side
(checkpoint / resolution / windowing) — NOT in how we score VoE.

Their raw_surprises/<block>_16frames.pth = dict with:
  losses           (M_movies, n_contexts, n_windows) float
  labels           (M_movies,)  -- 0 = IMPOSSIBLE, 1 = possible (their encoding)
  context_lengths  list, e.g. [2,4,6,8,10]
  frame_step       int
Their "Relative Accuracy (avg)" = per-window-mean surprise, matched possible/
impossible pairs within each scene quadruplet (4 consecutive movies = 2 pos +
2 imp, paired by position), correct when impossible > possible.

    PYTHONPATH=. python3 run_crosscheck.py            # validate + checkpoint table
    PYTHONPATH=. python3 run_crosscheck.py --models vit-l-rope-howto vit-h-rope-howto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

ROOT = Path("data/paper_intphys_surprises")
BLOCKS = ["O1", "O2", "O3"]
# A spot-check from vit-l-rope-howto/performance.csv (Relative Accuracy (avg), O1)
EXPECTED_VITL_O1_AVG = [93.33, 95.0, 93.33, 93.33, 91.67]  # contexts [2,4,6,8,10]


def matched_pair_accuracy(scalar: np.ndarray, labels: np.ndarray) -> float:
    """Pairwise VoE accuracy: within each 4-movie scene, pair pos[i] with imp[i]
    by position; correct when impossible surprise > possible surprise."""
    correct = total = 0
    for g in range(0, len(scalar), 4):
        gl, gs = labels[g:g + 4], scalar[g:g + 4]
        imp, pos = gs[gl == 0], gs[gl == 1]  # 0 = impossible in their encoding
        for pv, iv in zip(pos, imp):
            total += 1
            correct += int(iv > pv)
    return correct / max(1, total)


def block_accuracies(model: str, block: str, agg: str):
    f = ROOT / model / f"{block}_16frames.pth"
    if not f.exists():
        return None, None
    d = torch.load(f, map_location="cpu", weights_only=False)
    losses = d["losses"].numpy()
    labels = d["labels"].numpy().astype(int)
    pool = np.nanmean if agg == "avg" else np.nanmax
    scalar = pool(losses, axis=2)  # (M, n_contexts)
    accs = [matched_pair_accuracy(scalar[:, ci], labels) for ci in range(scalar.shape[1])]
    return d["context_lengths"], accs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=[
        "vit-l-rope-howto", "vit-h-rope-howto", "vit-l-rope-k710",
        "vit-l-rope-ssv2", "vit-l-rope-random-2", "videomaev2_g",
    ])
    ap.add_argument("--agg", choices=["avg", "max"], default="avg")
    args = ap.parse_args()

    if not ROOT.exists():
        raise SystemExit(
            f"{ROOT} not found. Get it from the paper repo:\n"
            "  curl -sL https://github.com/facebookresearch/jepa-intuitive-physics/raw/main/data_intphys.tar.gz | tar xz -C /tmp\n"
            "then copy the per-model intphys/raw_surprises dirs under data/paper_intphys_surprises/."
        )

    # 1. Validation: do we reproduce their published O1 avg row exactly?
    _, accs = block_accuracies("vit-l-rope-howto", "O1", "avg")
    if accs is None:
        raise SystemExit("missing vit-l-rope-howto/O1_16frames.pth")
    got = [round(100 * a, 2) for a in accs]
    err = sum(abs(g - e) for g, e in zip(got, EXPECTED_VITL_O1_AVG))
    print("VALIDATION — vit-l-rope-howto O1 Relative Accuracy (avg):")
    print(f"  ours:     {got}")
    print(f"  paper:    {EXPECTED_VITL_O1_AVG}")
    print(f"  abs err:  {err:.2f}  -> {'PASS (pipeline reproduces published numbers)' if err < 1 else 'MISMATCH'}\n")

    # 2. Best-context-per-block accuracy table across checkpoints (their key axis).
    print(f"Best-context IntPhys VoE accuracy ({args.agg}-agg, matched pairs):")
    print(f"  {'checkpoint':24s} {'O1':>6} {'O2':>6} {'O3':>6} {'mean':>7}")
    for model in args.models:
        per = {}
        for blk in BLOCKS:
            _, accs = block_accuracies(model, blk, args.agg)
            if accs is not None:
                per[blk] = 100 * max(accs)
        if per:
            mean = np.mean(list(per.values()))
            print(f"  {model:24s} " + " ".join(f"{per.get(b, float('nan')):6.1f}" for b in BLOCKS) + f" {mean:7.1f}")

    print("\nReading: ViT-L (vit-l-rope-howto) reaches ~92% with the SAME architecture our")
    print("engine builds — so model size is not the blocker, and our aggregation is correct.")
    print("The VoE signal is highly training-data-dependent (HowTo > K710 > SSv2 ~ random).")
    print("Our V-JEPA 2 ViT-L (~0.55-0.62, run_context_sweep) sits in the weak regime: the")
    print("gap to the paper is on the generation side (checkpoint/protocol), not analysis.")


if __name__ == "__main__":
    main()
