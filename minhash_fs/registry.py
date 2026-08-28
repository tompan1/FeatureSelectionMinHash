"""Name -> scorer registry used by the command-line experiment scripts."""

from __future__ import annotations

from functools import partial

from .proposed import kst_score, icd_max_score, ics_max_score, idis_score
from .baselines import (
    chi_squared_score,
    gain_ratio_score,
    gini_index_score,
    relieff_score,
    lasso_selection,
)

PROPOSED_METHODS = {
    "kst": kst_score,
    "icd_max": icd_max_score,
    "ics_max": ics_max_score,
    "idis": idis_score,           # alpha defaults to 0.5; swept in the scripts
}

SOTA_METHODS = {
    "chi_squared": chi_squared_score,
    "gain_ratio": gain_ratio_score,
    "gini_index": gini_index_score,
    "relieff": relieff_score,
    "lasso": lambda X, y: lasso_selection(X, y)[0],
}

FS_METHODS = {**PROPOSED_METHODS, **SOTA_METHODS}


def get_fs_method(name: str, **kwargs):
    """Return the scorer registered under ``name``, optionally pre-bound.

    Example
    -------
    >>> scorer = get_fs_method("idis", alpha=0.7)
    >>> scorer = get_fs_method("relieff", n_neighbors=10, n_samples_draw=500)
    """
    key = name.strip().lower().replace("-", "_")
    if key not in FS_METHODS:
        raise KeyError(f"Unknown FS method '{name}'. Available: {sorted(FS_METHODS)}")
    method = FS_METHODS[key]
    return partial(method, **kwargs) if kwargs else method
