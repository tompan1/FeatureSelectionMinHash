"""
The five state-of-the-art feature selection baselines used in the paper.

    chi-squared  Pearson (1900)      [21]
    gain ratio   Quinlan (1986)      [20]
    Gini index   Gini (1912)         [26]
    ReliefF      Kononenko (1994)    [22]
    Lasso (L1)   Tibshirani et al.   [25]

The first four are implemented directly on the nominal hash values, so that no
encoding step is required and the comparison against the proposed methods is
performed on exactly the same representation. Lasso is the only baseline that
requires a numeric interpretation of the hash values; the caveat is documented
in :func:`lasso_selection` and discussed in Section 7.B of the manuscript.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "chi_squared_score",
    "gain_ratio_score",
    "gini_index_score",
    "relieff_score",
    "lasso_selection",
]


def _contingency(col: np.ndarray, y_idx: np.ndarray, n_classes: int):
    """Contingency table of a nominal feature: rows = values, cols = classes."""
    values, inverse = np.unique(col, return_inverse=True)
    table = np.zeros((values.size, n_classes), dtype=np.float64)
    np.add.at(table, (inverse, y_idx), 1.0)
    return values, table


def _encode_labels(y):
    y = np.asarray(y)
    classes = np.unique(y)
    index = {c: j for j, c in enumerate(classes)}
    y_idx = np.fromiter((index[v] for v in y), dtype=np.int64, count=y.shape[0])
    return classes, y_idx


# --------------------------------------------------------------------------- #
# Chi-squared
# --------------------------------------------------------------------------- #
def chi_squared_score(X, y) -> np.ndarray:
    """Pearson chi-squared statistic between each nominal feature and the label."""
    X = np.asarray(X)
    classes, y_idx = _encode_labels(y)
    n_samples = X.shape[0]
    class_totals = np.bincount(y_idx, minlength=classes.size).astype(np.float64)

    scores = np.zeros(X.shape[1], dtype=float)
    for i in range(X.shape[1]):
        _, observed = _contingency(X[:, i], y_idx, classes.size)
        row_totals = observed.sum(axis=1, keepdims=True)
        expected = row_totals @ class_totals[None, :] / n_samples
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)
        scores[i] = terms.sum()
    return scores


# --------------------------------------------------------------------------- #
# Information gain ratio
# --------------------------------------------------------------------------- #
def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def gain_ratio_score(X, y) -> np.ndarray:
    """Quinlan's gain ratio = information gain / intrinsic value (split info)."""
    X = np.asarray(X)
    classes, y_idx = _encode_labels(y)
    n_samples = X.shape[0]
    parent_entropy = _entropy(np.bincount(y_idx, minlength=classes.size))

    scores = np.zeros(X.shape[1], dtype=float)
    for i in range(X.shape[1]):
        _, table = _contingency(X[:, i], y_idx, classes.size)
        row_totals = table.sum(axis=1)
        weights = row_totals / n_samples
        conditional = sum(w * _entropy(row) for w, row in zip(weights, table))
        info_gain = parent_entropy - conditional
        intrinsic = _entropy(row_totals)
        scores[i] = info_gain / intrinsic if intrinsic > 0 else 0.0
    return scores


# --------------------------------------------------------------------------- #
# Gini index
# --------------------------------------------------------------------------- #
def gini_index_score(X, y) -> np.ndarray:
    """Gini impurity decrease obtained by splitting on the nominal feature.

    Higher values indicate a feature that separates the classes better.
    """
    X = np.asarray(X)
    classes, y_idx = _encode_labels(y)
    n_samples = X.shape[0]
    priors = np.bincount(y_idx, minlength=classes.size) / n_samples
    parent_gini = 1.0 - np.sum(priors ** 2)

    scores = np.zeros(X.shape[1], dtype=float)
    for i in range(X.shape[1]):
        _, table = _contingency(X[:, i], y_idx, classes.size)
        row_totals = table.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            proportions = np.where(row_totals > 0, table / row_totals, 0.0)
        child_gini = 1.0 - np.sum(proportions ** 2, axis=1)
        weighted = np.sum((row_totals.ravel() / n_samples) * child_gini)
        scores[i] = parent_gini - weighted
    return scores


# --------------------------------------------------------------------------- #
# ReliefF
# --------------------------------------------------------------------------- #
def relieff_score(X, y, n_neighbors: int = 10, n_samples_draw: int | None = None,
                  random_state: int | None = 42) -> np.ndarray:
    """ReliefF (Kononenko, 1994) with the nominal distance diff(a, b) = [a != b].

    For each of ``n_samples_draw`` randomly drawn instances, the ``k`` nearest
    hits (same class) and the ``k`` nearest misses of every other class are
    located under the Hamming distance over the hash values; misses are weighted
    by their class prior, as prescribed by ReliefF.

    Parameters
    ----------
    n_samples_draw : int, optional
        Number of instances drawn (``m`` in the original paper). ``None`` uses
        every instance, which is exact but O(N^2 * M).
    """
    X = np.asarray(X)
    rng = np.random.default_rng(random_state)
    classes, y_idx = _encode_labels(y)
    n_samples, n_features = X.shape
    priors = np.bincount(y_idx, minlength=classes.size) / n_samples

    if n_samples_draw is None or n_samples_draw >= n_samples:
        drawn = np.arange(n_samples)
    else:
        drawn = rng.choice(n_samples, size=n_samples_draw, replace=False)

    weights = np.zeros(n_features, dtype=float)
    class_members = [np.flatnonzero(y_idx == c) for c in range(classes.size)]

    for idx in drawn:
        diff_matrix = (X != X[idx]).astype(np.float64)   # (n_samples, n_features)
        distances = diff_matrix.sum(axis=1)
        distances[idx] = np.inf                          # never select itself
        own = y_idx[idx]

        # nearest hits
        hits = class_members[own]
        hits = hits[hits != idx]
        if hits.size:
            k = min(n_neighbors, hits.size)
            nearest = hits[np.argpartition(distances[hits], k - 1)[:k]]
            weights -= diff_matrix[nearest].mean(axis=0)

        # nearest misses, weighted by the class prior
        denom = 1.0 - priors[own]
        if denom <= 0:
            continue
        for c in range(classes.size):
            if c == own:
                continue
            members = class_members[c]
            if members.size == 0:
                continue
            k = min(n_neighbors, members.size)
            nearest = members[np.argpartition(distances[members], k - 1)[:k]]
            weights += (priors[c] / denom) * diff_matrix[nearest].mean(axis=0)

    return weights / max(len(drawn), 1)


# --------------------------------------------------------------------------- #
# Lasso (L1 regularisation)
# --------------------------------------------------------------------------- #
def lasso_selection(X, y, C: float = 1.0, random_state: int | None = 42,
                    max_iter: int = 2000):
    """L1-regularised logistic regression; features with a non-zero coefficient
    are retained.

    Caveat (Section 7.B of the manuscript): Lasso is a linear model and
    therefore requires the hash values to be read as numbers, which contradicts
    their nominal nature. It is included precisely because it is a standard
    choice for n-gram features, and the manuscript reports that it is largely
    ineffective on MinHash features. The values are standardised before fitting
    so that the penalty is applied on a comparable scale.

    Returns
    -------
    scores : ndarray
        Absolute coefficient magnitude per feature (0 for discarded features).
    selected : ndarray
        Indices of the features with a non-zero coefficient.
    """
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    solver = "liblinear" if len(np.unique(y)) == 2 else "saga"
    common = dict(C=C, max_iter=max_iter, random_state=random_state, solver=solver)
    # scikit-learn >= 1.8 deprecates `penalty` in favour of `l1_ratio`
    if tuple(int(part) for part in sklearn.__version__.split(".")[:2]) >= (1, 8):
        model = LogisticRegression(l1_ratio=1.0, **common)
    else:
        model = LogisticRegression(penalty="l1", **common)
    model.fit(Xs, y)
    coef = np.abs(model.coef_).max(axis=0)
    selected = np.flatnonzero(coef > 0)
    return coef, selected
