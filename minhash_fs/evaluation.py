"""
The evaluation protocol of Section 7 (Experiments 1 and 2).

Protocol
--------
1. A MinHash sketch of 400 features is built for every item (Experiment 1) or
   two 200-feature sketches are concatenated (Experiment 2).
2. Each FS method ranks the 400 features.
3. The classifier is retrained on the top-k features for
   k = 10, 20, ..., 390 (and on all 400 features as the baseline).
4. Performance is the mean over a stratified 10-fold cross-validation:
   AUC for binary tasks, accuracy for the multi-class task.
5. For every FS method the best point of the sweep is reported, together with
   the number of features at which it was reached.

Feature-selection scope
-----------------------
``fs_scope`` controls where the feature ranking is computed:

``"train_fold"`` (default)
    the ranking is recomputed inside each training fold, so no information from
    the held-out fold reaches the selection step. This is the leakage-free
    protocol and the one we recommend for any new comparison.
``"full"``
    the ranking is computed once on the whole dataset and reused in every fold.
    It is faster and is provided for exact comparability with published results
    that follow this convention.

Both scopes are supported end to end; the scope actually used is written into
the results file so that any reported number is unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .classifiers import build_classifier
from .proposed import rank_features

__all__ = ["FoldResult", "SweepResult", "cross_validate", "feature_sweep",
           "default_k_grid"]


def default_k_grid(n_features: int = 400, step: int = 10) -> list[int]:
    """k = 10, 20, ..., n_features - step  (the grid of Tables 2 and 3)."""
    return list(range(step, n_features, step))


# --------------------------------------------------------------------------- #
@dataclass
class FoldResult:
    """Per-fold scores of one (classifier, feature subset) configuration."""
    scores: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.scores)) if self.scores else float("nan")

    @property
    def std(self) -> float:
        return float(np.std(self.scores, ddof=1)) if len(self.scores) > 1 else 0.0


@dataclass
class SweepResult:
    """Outcome of a full top-k sweep for one FS method and one classifier."""
    method: str
    classifier: str
    metric: str
    fs_scope: str
    k_values: list[int]
    mean_scores: list[float]
    fold_scores: dict            # k -> list of per-fold scores

    @property
    def best_k(self) -> int:
        return int(self.k_values[int(np.argmax(self.mean_scores))])

    @property
    def best_score(self) -> float:
        return float(np.max(self.mean_scores))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["best_k"] = self.best_k
        d["best_score"] = self.best_score
        return d


# --------------------------------------------------------------------------- #
def _score_predictions(clf, X_test, y_test, classes, metric: str) -> float:
    if metric == "auc":
        proba = clf.predict_proba(X_test)
        if len(classes) == 2:
            return float(roc_auc_score(y_test, proba[:, 1]))
        return float(roc_auc_score(y_test, proba, multi_class="ovr",
                                   average="macro", labels=classes))
    if metric == "accuracy":
        return float(accuracy_score(y_test, clf.predict(X_test)))
    raise ValueError(f"Unknown metric '{metric}'")


def cross_validate(X, y, classifier: str = "sc", metric: str = "auc",
                   n_splits: int = 10, random_state: int = 42,
                   feature_indices: Sequence[int] | None = None) -> FoldResult:
    """Stratified k-fold cross-validation of one classifier on a fixed subset."""
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    if feature_indices is not None:
        X = X[:, np.asarray(feature_indices)]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    result = FoldResult()
    for train_idx, test_idx in splitter.split(X, y):
        clf = build_classifier(classifier, random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        result.scores.append(
            _score_predictions(clf, X[test_idx], y[test_idx], classes, metric)
        )
    return result


def feature_sweep(X, y, scorer: Callable, method_name: str,
                  classifier: str = "sc", metric: str = "auc",
                  k_values: Sequence[int] | None = None,
                  n_splits: int = 10, random_state: int = 42,
                  fs_scope: str = "train_fold",
                  verbose: bool = False) -> SweepResult:
    """Evaluate a classifier over the top-k features of one FS method.

    Parameters
    ----------
    scorer : callable
        ``scorer(X, y) -> array of feature scores`` (higher is better).
    fs_scope : {"train_fold", "full"}
        See the module docstring.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    n_features = X.shape[1]
    k_values = list(k_values) if k_values is not None else default_k_grid(n_features)
    k_values = [k for k in k_values if 1 <= k <= n_features]

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=random_state)
    splits = list(splitter.split(X, y))

    if fs_scope == "full":
        global_ranking = rank_features(scorer(X, y))
        fold_rankings = [global_ranking] * len(splits)
    elif fs_scope == "train_fold":
        fold_rankings = [rank_features(scorer(X[tr], y[tr])) for tr, _ in splits]
    else:
        raise ValueError("fs_scope must be 'train_fold' or 'full'")

    fold_scores = {k: [] for k in k_values}
    for (train_idx, test_idx), ranking in zip(splits, fold_rankings):
        for k in k_values:
            columns = np.sort(ranking[:k])
            clf = build_classifier(classifier, random_state=random_state)
            clf.fit(X[np.ix_(train_idx, columns)], y[train_idx])
            fold_scores[k].append(
                _score_predictions(clf, X[np.ix_(test_idx, columns)],
                                   y[test_idx], classes, metric)
            )
        if verbose:
            print(f"  [{method_name}/{classifier}] fold done", flush=True)

    mean_scores = [float(np.mean(fold_scores[k])) for k in k_values]
    return SweepResult(method=method_name, classifier=classifier, metric=metric,
                       fs_scope=fs_scope, k_values=k_values,
                       mean_scores=mean_scores, fold_scores=fold_scores)
