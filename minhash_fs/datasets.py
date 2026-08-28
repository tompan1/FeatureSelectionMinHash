"""
Loading and saving of MinHash feature matrices.

A MinHash dataset is stored as a CSV (or compressed ``.npz``) file with one row
per item, columns ``f0 ... f{k-1}`` holding the hash values, and a final column
``label``. This is the only format the experiment scripts need, which keeps the
feature selection code independent of the raw data modality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_minhash_csv", "save_minhash_csv", "load_minhash_npz",
           "save_minhash_npz", "toy_example"]


def save_minhash_csv(path, X, y, label_column: str = "label") -> Path:
    """Write a MinHash matrix and its labels to CSV."""
    X = np.asarray(X)
    y = np.asarray(y)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ",".join([f"f{i}" for i in range(X.shape[1])] + [label_column])
    rows = np.column_stack([X, y])
    np.savetxt(path, rows, delimiter=",", header=header, comments="", fmt="%s")
    return path


def load_minhash_csv(path, label_column: str = "label"):
    """Read a MinHash CSV written by :func:`save_minhash_csv`.

    Returns
    -------
    X : ndarray of int64, shape (n_items, n_features)
    y : ndarray, shape (n_items,)
    """
    path = Path(path)
    with path.open() as handle:
        header = handle.readline().strip().split(",")
    if label_column not in header:
        raise ValueError(f"column '{label_column}' not found in {path}")
    label_idx = header.index(label_column)
    feature_idx = [i for i in range(len(header)) if i != label_idx]

    raw = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    if raw.ndim == 1:
        raw = raw[None, :]
    X = raw[:, feature_idx].astype(np.int64)
    y = raw[:, label_idx]
    try:                       # keep integer labels integral where possible
        y = y.astype(np.int64)
    except ValueError:
        pass
    return X, y


def save_minhash_npz(path, X, y) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=np.asarray(X), y=np.asarray(y))
    return path


def load_minhash_npz(path):
    data = np.load(Path(path), allow_pickle=True)
    return data["X"], data["y"]


# --------------------------------------------------------------------------- #
def toy_example():
    """The illustrative dataset of Figure 1 (Section 3.D).

    Five features over eight samples of two classes, covering the three extreme
    cases and the two common ones:

        f0  one hash value shared by every sample        - no discriminative power
        f1  one distinct hash value per class            - perfect separation
        f2  a unique hash value for every sample         - unusable cardinality
        f3  values repeat inside a class, some overlap   - typical case
        f4  many values per class, repeats proportional
            to the similarity between samples            - typical of large data

    Expected usefulness, as argued in the paper:
        f1 > f4 > f3 > f2 > f0

    The values below reconstruct the *structure* described in Figure 1; they are
    not a transcription of the exact hash values plotted there. Substitute the
    published values if the figure numbers are to be reproduced verbatim.
    """
    X = np.array([
        # f0    f1     f2     f3     f4
        [1000, 2001, 3001, 4001, 5001],
        [1000, 2001, 3002, 4001, 5002],
        [1000, 2001, 3003, 4002, 5001],
        [1000, 2001, 3004, 4002, 5003],
        [1000, 2002, 3005, 4002, 5004],
        [1000, 2002, 3006, 4003, 5005],
        [1000, 2002, 3007, 4003, 5004],
        [1000, 2002, 3008, 4001, 5006],
    ], dtype=np.int64)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
    return X, y, feature_names
