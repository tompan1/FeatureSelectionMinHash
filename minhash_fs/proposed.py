"""
The four feature selection methods proposed in the paper.

======================================================================
Method     Manuscript reference                     Complexity
----------------------------------------------------------------------
KST        Algorithm 1 (Kolmogorov-Smirnov test)    O(M*N log N)  binary
                                                    O(M*C*N log N) multi-class
ICD-Max    Algorithm 2 (inter-class dissimilarity)  O(M*N) / O(M*C*N)
ICS-Max    Algorithm 3 (intra-class similarity)     O(M*N)
IDIS       Equation 9 (weighted combination)        O(M*C*N)
======================================================================

Every scorer takes an integer feature matrix ``X`` of shape
``(n_samples, n_features)`` holding MinHash values, and a label vector ``y``.
Every scorer returns a float vector of length ``n_features``; **higher is
better**, and features are ranked in descending order of their score.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "class_weights",
    "ks_statistic",
    "kst_score",
    "icd_max_score",
    "ics_max_score",
    "idis_score",
    "rank_features",
    "select_top_k",
]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _as_matrix(X) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_samples, n_features), got shape {X.shape}")
    return X


def class_weights(y) -> dict:
    """Weight vector W of Algorithms 1-2: w_c = |class c| / |samples|."""
    y = np.asarray(y)
    classes, counts = np.unique(y, return_counts=True)
    return {c: n / y.shape[0] for c, n in zip(classes, counts)}


def rank_features(scores) -> np.ndarray:
    """Indices of the features ordered by descending score (best first).

    Ties are broken by feature index so that the ranking is deterministic and
    reproducible across runs and platforms.
    """
    scores = np.asarray(scores, dtype=float)
    return np.lexsort((np.arange(scores.shape[0]), -scores))


def select_top_k(X, scores, k: int) -> np.ndarray:
    """Return ``X`` restricted to the ``k`` highest-scoring features."""
    order = rank_features(scores)[:k]
    return _as_matrix(X)[:, np.sort(order)]


# --------------------------------------------------------------------------- #
# Algorithm 1 - Kolmogorov-Smirnov test (KST)
# --------------------------------------------------------------------------- #
def ks_statistic(a, b) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D = max |CDF_a(x) - CDF_b(x)|.

    Implemented directly in NumPy (sort + merge) so that the package has no
    hard SciPy dependency. The value is identical to
    ``scipy.stats.ks_2samp(a, b).statistic``.
    """
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return 0.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


def kst_score(X, y) -> np.ndarray:
    """Algorithm 1 - rank features by the distributional distance between classes.

    Binary task
        D_i = KS(CDF of the hash values of the positive class,
                 CDF of the hash values of the negative class)

    Multi-class task (one-vs-rest, weighted by class prior)
        D_i = sum_c w_c * KS(CDF_c, CDF_{not c})

    Note on the use of an ordinal test over nominal data: the KST is applied to
    the *hash domain ordering* only. No arithmetic is performed on the hash
    values themselves; the test compares the shape of the two empirical
    distributions over that fixed ordering, which is well defined for any
    totally ordered domain (Section 6.A of the paper).
    """
    X = _as_matrix(X)
    y = np.asarray(y)
    classes = np.unique(y)
    n_features = X.shape[1]
    scores = np.zeros(n_features, dtype=float)

    if classes.size < 2:
        raise ValueError("At least two classes are required for feature selection.")

    if classes.size == 2:
        pos_mask = y == classes[1]
        neg_mask = ~pos_mask
        for i in range(n_features):
            scores[i] = ks_statistic(X[pos_mask, i], X[neg_mask, i])
        return scores

    weights = class_weights(y)
    for c in classes:
        in_c = y == c
        w_c = weights[c]
        for i in range(n_features):
            scores[i] += w_c * ks_statistic(X[in_c, i], X[~in_c, i])
    return scores


# --------------------------------------------------------------------------- #
# Algorithm 2 - Inter-Class Dissimilarity Maximization (ICD-Max)
# --------------------------------------------------------------------------- #
def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity coefficient; J(empty, empty) = 1 (Section 3.A)."""
    union = len(set_a | set_b)
    if union == 0:
        return 1.0
    return len(set_a & set_b) / union


def icd_max_score(X, y) -> np.ndarray:
    """Algorithm 2 - prefer features whose classes use disjoint sets of hashes.

    Binary task
        score_i = 1 - J(set of hashes of class +, set of hashes of class -)

    Multi-class task (one-vs-rest, weighted by class prior)
        score_i = 1 - sum_c w_c * J(set of hashes of class c,
                                    set of hashes of all other classes)
    """
    X = _as_matrix(X)
    y = np.asarray(y)
    classes = np.unique(y)
    n_features = X.shape[1]
    scores = np.zeros(n_features, dtype=float)

    if classes.size < 2:
        raise ValueError("At least two classes are required for feature selection.")

    if classes.size == 2:
        pos_mask = y == classes[1]
        neg_mask = ~pos_mask
        for i in range(n_features):
            col = X[:, i]
            scores[i] = 1.0 - _jaccard(set(col[pos_mask].tolist()),
                                       set(col[neg_mask].tolist()))
        return scores

    weights = class_weights(y)
    weighted_similarity = np.zeros(n_features, dtype=float)
    for c in classes:
        in_c = y == c
        w_c = weights[c]
        for i in range(n_features):
            col = X[:, i]
            weighted_similarity[i] += w_c * _jaccard(set(col[in_c].tolist()),
                                                     set(col[~in_c].tolist()))
    return 1.0 - weighted_similarity


# --------------------------------------------------------------------------- #
# Algorithm 3 - Intra-Class Similarity Maximization (ICS-Max)
# --------------------------------------------------------------------------- #
def ics_max_score(X, y, multiclass_rule: str = "argmax") -> np.ndarray:
    """Algorithm 3 - reward hash values that repeat inside a single class.

    A hash value contributes to the score of its feature only if
    (i) it occurs more than once in that feature, and
    (ii) it is strictly more frequent in one class than elsewhere.
    In that case its occurrences inside the dominant class are accumulated.
    The feature score is the accumulated count divided by the number of samples,
    i.e. the fraction of "beneficial" hash occurrences.

    Parameters
    ----------
    multiclass_rule : {"argmax", "vs_rest"}
        Only relevant for multi-class tasks; the two rules coincide for binary
        tasks.

        ``"argmax"`` (default, matching the prose of Section 6.C)
            the value is credited to the class in which it appears most
            frequently, provided that class is unique.
        ``"vs_rest"`` (matching the pseudo-code of Algorithm 3 literally)
            the value is credited only if its count in the most frequent class
            strictly exceeds its total count in *all* the remaining classes.
    """
    if multiclass_rule not in ("argmax", "vs_rest"):
        raise ValueError("multiclass_rule must be 'argmax' or 'vs_rest'")

    X = _as_matrix(X)
    y = np.asarray(y)
    classes = np.unique(y)
    n_samples, n_features = X.shape
    class_index = {c: j for j, c in enumerate(classes)}
    y_idx = np.fromiter((class_index[v] for v in y), dtype=np.int64, count=n_samples)

    scores = np.zeros(n_features, dtype=float)
    for i in range(n_features):
        col = X[:, i]
        values, inverse = np.unique(col, return_inverse=True)
        # counts[v, c] = number of occurrences of value v inside class c
        counts = np.zeros((values.size, classes.size), dtype=np.int64)
        np.add.at(counts, (inverse, y_idx), 1)
        totals = counts.sum(axis=1)

        repeated = totals > 1
        if not np.any(repeated):
            continue

        cnt = counts[repeated]
        tot = totals[repeated]
        best = cnt.max(axis=1)
        # a dominant class must be unique (ties contribute nothing)
        unique_max = (cnt == best[:, None]).sum(axis=1) == 1
        if multiclass_rule == "argmax":
            keep = unique_max
        else:  # "vs_rest": count in the best class must beat all the others
            keep = unique_max & (best > (tot - best))
        scores[i] = best[keep].sum() / n_samples

    return scores


# --------------------------------------------------------------------------- #
# Equation 9 - Inter Dissimilarity Intra Similarity (IDIS)
# --------------------------------------------------------------------------- #
def idis_score(X, y, alpha: float = 0.5, multiclass_rule: str = "argmax") -> np.ndarray:
    """Equation 9 - weighted combination of ICD-Max and ICS-Max.

        score_IDIS = alpha * score_ICD-Max + (1 - alpha) * score_ICS-Max

    ``alpha > 0.5`` prioritises inter-class discriminative power,
    ``alpha < 0.5`` prioritises intra-class consistency.
    Both component scores already live on the [0, 1] scale, so no additional
    normalisation is applied.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    icd = icd_max_score(X, y)
    ics = ics_max_score(X, y, multiclass_rule=multiclass_rule)
    return alpha * icd + (1.0 - alpha) * ics


def idis_score_from_components(icd, ics, alpha: float = 0.5) -> np.ndarray:
    """IDIS score from pre-computed ICD-Max / ICS-Max scores.

    Useful when sweeping over several values of ``alpha``: the two component
    scores are computed once and only the combination is repeated.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    return alpha * np.asarray(icd, dtype=float) + (1.0 - alpha) * np.asarray(ics, dtype=float)
