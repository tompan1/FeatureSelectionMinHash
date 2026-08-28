"""
MinHash sketch construction (Section 3.B, Equations 4-7).

A sketch of size k is obtained by applying k independent Rabin fingerprint
functions to every shingle of an item and keeping, for each function, the
minimum fingerprint:

    M(D)[j] = min { pi_j(s) : s in S(D, w) },     j = 1 .. k

The resulting vector of k hash values is the feature vector of the item. Two
items are compared by the fraction of positions in which their sketches agree,
which is an unbiased estimator of the Jaccard similarity of their shingle sets.

With k = 400 the estimation error of the Jaccard similarity is bounded by
roughly 1/sqrt(k) = 0.05, which is the configuration used in the paper.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from .rabin import RabinFingerprint, make_fingerprint_family
from .shingles import extract_shingles, image_partitions

__all__ = ["minhash_sketch", "build_sketch_matrix", "partition_minhash_sketch",
           "combine_sketches"]

EMPTY_SENTINEL = -1      # marks an item that produced no shingle at all


def minhash_sketch(shingles: Iterable[bytes],
                   family: Sequence[RabinFingerprint]) -> np.ndarray:
    """MinHash sketch of one item, given its shingles and a fingerprint family."""
    k = len(family)
    minima = np.full(k, np.iinfo(np.int64).max, dtype=np.int64)
    seen_any = False
    for shingle in shingles:
        seen_any = True
        for j, fingerprint in enumerate(family):
            value = fingerprint(shingle)
            if value < minima[j]:
                minima[j] = value
    if not seen_any:
        minima[:] = EMPTY_SENTINEL
    return minima


def partition_minhash_sketch(image: np.ndarray,
                             family: Sequence[RabinFingerprint],
                             window: int, overlap: int,
                             n_rows: int = 2, n_cols: int = 2) -> np.ndarray:
    """Partition MinHash [4] for images: the family is split across partitions.

    The k functions are distributed evenly over the ``n_rows * n_cols``
    partitions, so the concatenated sketch still has exactly k entries while
    each entry summarises one image region.
    """
    partitions = list(image_partitions(image, n_rows=n_rows, n_cols=n_cols))
    chunks = np.array_split(np.arange(len(family)), len(partitions))
    sketch = np.empty(len(family), dtype=np.int64)
    for partition, chunk in zip(partitions, chunks):
        sub_family = [family[j] for j in chunk]
        shingles = extract_shingles(partition, "image_2d", window, overlap)
        sketch[chunk] = minhash_sketch(shingles, sub_family)
    return sketch


def build_sketch_matrix(items: Iterable, modality: str, length: int, overlap: int,
                        n_functions: int = 400, degree: int = 61,
                        random_state: int | None = 42,
                        verbose: bool = False) -> np.ndarray:
    """Sketch every item in ``items`` and stack the results.

    The same fingerprint family (fixed by ``random_state``) is used for every
    item and, in the paper, for every dataset.

    Returns
    -------
    ndarray of shape (n_items, n_functions), dtype int64
    """
    family = make_fingerprint_family(n_functions, degree=degree,
                                     random_state=random_state)
    rows = []
    for index, item in enumerate(items):
        rows.append(minhash_sketch(extract_shingles(item, modality, length, overlap),
                                   family))
        if verbose and (index + 1) % 100 == 0:
            print(f"  sketched {index + 1} items", flush=True)
    return np.vstack(rows) if rows else np.empty((0, n_functions), dtype=np.int64)


def combine_sketches(*sketches: np.ndarray) -> np.ndarray:
    """Feature aggregation of Experiment 2: concatenate sketches of one item set.

    In the paper two sketches of 200 features each (same shingle size, different
    overlap) are concatenated into a single 400-feature vector, on which feature
    selection is then applied.
    """
    if not sketches:
        raise ValueError("at least one sketch matrix is required")
    n_rows = {s.shape[0] for s in sketches}
    if len(n_rows) != 1:
        raise ValueError("all sketch matrices must describe the same items")
    return np.hstack(sketches)
