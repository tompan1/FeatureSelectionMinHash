"""
The three classifiers used in the experiments.

    SC       Similarity Classifier - assigns a test item to the class with the
             highest average MinHash (Jaccard) similarity. It is a k-NN variant
             in which the metric is derived from the Jaccard similarity, which
             makes it the natural classifier for MinHash representations.
    RF       Random Forest, 100 trees, unpruned.
    XGBoost  100 boosting rounds, fixed learning rate, bounded depth.

Hyper-parameters are deliberately *not* tuned: they are identical in every
experiment so that any difference in performance can be attributed to the
feature selection method alone (Section 7.B).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

__all__ = ["SimilarityClassifier", "minhash_similarity", "build_classifier",
           "CLASSIFIER_NAMES"]

CLASSIFIER_NAMES = ("sc", "rf", "xgb")


def minhash_similarity(A: np.ndarray, B: np.ndarray, block_size: int = 256) -> np.ndarray:
    """Estimated Jaccard similarity between every row of A and every row of B.

    The estimator is the fraction of sketch positions holding the same hash
    value (Equation 7). Computed in blocks to bound the memory footprint.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float32)
    for start in range(0, A.shape[0], block_size):
        stop = min(start + block_size, A.shape[0])
        matches = (A[start:stop, None, :] == B[None, :, :])
        out[start:stop] = matches.mean(axis=2, dtype=np.float32)
    return out


class SimilarityClassifier(BaseEstimator, ClassifierMixin):
    """Average-similarity classifier over MinHash sketches.

    For a test item, the mean estimated Jaccard similarity to the training items
    of each class is computed; the predicted class is the one with the highest
    mean similarity. The decision score used for AUC is the normalised class
    similarity, so the classifier is threshold-free by construction.

    Parameters
    ----------
    block_size : int
        Number of test rows processed per block; lower it if memory is tight.
    """

    def __init__(self, block_size: int = 256):
        self.block_size = block_size

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.X_ = X
        self.y_ = y
        self._masks_ = [y == c for c in self.classes_]
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        similarity = minhash_similarity(X, self.X_, block_size=self.block_size)
        means = np.column_stack([similarity[:, mask].mean(axis=1) for mask in self._masks_])
        totals = means.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            proba = np.where(totals > 0, means / totals, 1.0 / len(self.classes_))
        return proba.astype(np.float64)

    def decision_function(self, X):
        proba = self.predict_proba(X)
        return proba[:, 1] if len(self.classes_) == 2 else proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_classifier(name: str, random_state: int = 42, n_jobs: int = -1):
    """Instantiate one of the three classifiers with the parameters of the paper."""
    key = name.strip().lower()
    if key == "sc":
        return SimilarityClassifier()
    if key == "rf":
        # 100 trees, no pruning (fully grown trees)
        return RandomForestClassifier(
            n_estimators=100, max_depth=None, min_samples_split=2,
            min_samples_leaf=1, random_state=random_state, n_jobs=n_jobs,
        )
    if key == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:      # pragma: no cover - environment dependent
            raise ImportError(
                "xgboost is required for the XGB classifier: pip install xgboost"
            ) from exc
        return XGBClassifier(
            n_estimators=100, learning_rate=0.3, max_depth=6,
            tree_method="hist", random_state=random_state, n_jobs=n_jobs,
            eval_metric="logloss",
        )
    raise ValueError(f"Unknown classifier '{name}'. Expected one of {CLASSIFIER_NAMES}")
