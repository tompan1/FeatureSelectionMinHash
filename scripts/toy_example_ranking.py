#!/usr/bin/env python3
"""
Ranking behaviour of every FS method on the toy dataset of Section 3.D.

The toy dataset contains five features covering the three extreme cases and the
two common ones described in Figure 1. The paper argues that the expected order
of usefulness is

    Feature 2 > Feature 5 > Feature 4 > Feature 3 > Feature 1

and then shows that chi-squared, the gain ratio and ReliefF each depart from it
in a characteristic way. This script prints the ranking produced by every FS
method, so that the argument of Section 3.D can be inspected and re-run.

NOTE: the hash values in ``minhash_fs.datasets.toy_example`` are a faithful
reconstruction of the *structure* described in Figure 1, not a transcription of
the exact values plotted there. Replace them with the published values to obtain
the figure numbers verbatim.

Usage
-----
    python scripts/toy_example_ranking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minhash_fs.datasets import toy_example
from minhash_fs.proposed import rank_features
from minhash_fs.registry import PROPOSED_METHODS, SOTA_METHODS


def main() -> None:
    X, y, names = toy_example()
    print("Toy dataset of Figure 1 "
          f"({X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(np.unique(y))} classes)\n")
    print("Expected order of usefulness (Section 3.D):")
    print("  Feature 2 > Feature 5 > Feature 4 > Feature 3 > Feature 1\n")

    methods = {**PROPOSED_METHODS}
    methods.pop("idis")                       # reported separately, per alpha
    methods.update({k: v for k, v in SOTA_METHODS.items() if k != "lasso"})

    header = f"{'method':<14} ranking (best -> worst)"
    print(header)
    print("-" * len(header))
    for name, scorer in methods.items():
        scores = np.asarray(scorer(X, y), dtype=float)
        order = rank_features(scores)
        ranking = " > ".join(names[i] for i in order)
        print(f"{name:<14} {ranking}")
        print(f"{'':<14} scores: " +
              ", ".join(f"{names[i]}={scores[i]:.3f}" for i in range(len(names))))

    print("\nIDIS over the alpha grid of Equation 9:")
    from minhash_fs.proposed import icd_max_score, ics_max_score, idis_score_from_components
    icd, ics = icd_max_score(X, y), ics_max_score(X, y)
    for alpha in (0.3, 0.4, 0.5, 0.6, 0.7):
        scores = idis_score_from_components(icd, ics, alpha=alpha)
        order = rank_features(scores)
        print(f"  alpha={alpha:<4} " + " > ".join(names[i] for i in order))


if __name__ == "__main__":
    main()
