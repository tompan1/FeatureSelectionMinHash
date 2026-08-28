#!/usr/bin/env python3
"""
Generate a small synthetic MinHash dataset.

The malware samples underlying the dynamic-analysis dataset cannot be
redistributed, and the remaining four datasets are large third-party downloads
(see data/README.md). This script produces a synthetic MinHash matrix with the
same *structure* as a real sketch - nominal, independent, high-cardinality hash
columns with class-dependent collision rates - so that the whole pipeline can be
executed and unit-tested without any restricted data.

The synthetic data are for smoke-testing and demonstration only; no result
reported in the paper is derived from them.

Usage
-----
    python scripts/make_demo_dataset.py --out data/demo/demo_binary.csv
    python scripts/make_demo_dataset.py --classes 5 --metric-hint accuracy \
        --out data/demo/demo_multiclass.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minhash_fs.datasets import save_minhash_csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=400)
    p.add_argument("--features", type=int, default=60)
    p.add_argument("--classes", type=int, default=2)
    p.add_argument("--informative-fraction", type=float, default=0.3,
                   help="fraction of features carrying class signal")
    p.add_argument("--vocabulary", type=int, default=40,
                   help="distinct hash values per feature (cardinality)")
    p.add_argument("--leak", type=float, default=0.45,
                   help="probability that an informative feature draws from the "
                        "shared pool, i.e. how much the classes overlap")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/demo/demo_binary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    n_informative = max(1, int(round(args.features * args.informative_fraction)))
    y = rng.integers(0, args.classes, size=args.samples)
    X = np.empty((args.samples, args.features), dtype=np.int64)

    # a large offset per feature mimics an independent hash domain per function
    offsets = rng.integers(10 ** 9, 2 * 10 ** 9, size=args.features)

    for j in range(args.features):
        if j < n_informative:
            # informative: each class draws from a mostly private pool of hashes
            pool = args.vocabulary // args.classes
            values = np.empty(args.samples, dtype=np.int64)
            for c in range(args.classes):
                mask = y == c
                private = rng.integers(c * pool, (c + 1) * pool, size=int(mask.sum()))
                # some draws leak into the shared pool, so the classes overlap
                leak = rng.random(int(mask.sum())) < args.leak
                shared = rng.integers(0, args.vocabulary, size=int(mask.sum()))
                values[mask] = np.where(leak, shared, private)
            X[:, j] = offsets[j] + values
        else:
            # non-informative: identical distribution in every class
            X[:, j] = offsets[j] + rng.integers(0, args.vocabulary, size=args.samples)

    save_minhash_csv(args.out, X, y)
    print(f"Wrote synthetic dataset: {args.samples} samples x {args.features} features, "
          f"{args.classes} classes, {n_informative} informative -> {args.out}")


if __name__ == "__main__":
    main()
