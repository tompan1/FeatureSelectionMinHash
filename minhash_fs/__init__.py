"""
minhash_fs
==========

Reference implementation of the feature selection (FS) methods proposed in:

    T. Panker and N. Nissim, "Feature Selection Strategies for Improving
    MinHash-Based Classifiers' Performance and Generalization".

The package contains:

* ``proposed``  - the four proposed FS methods (KST, ICD-Max, ICS-Max, IDIS)
* ``baselines`` - the five SOTA baselines used for comparison
                  (chi-squared, gain ratio, Gini index, ReliefF, Lasso)
* ``minhash``   - Rabin fingerprinting, shingling and MinHash sketch generation
* ``classifiers`` - the Similarity Classifier (SC) used in the paper
* ``evaluation``  - the iterative top-k / 10-fold cross-validation protocol
* ``stats``       - the one-sided paired t-test used for significance testing

All FS methods share the same interface::

    scores = method(X, y)          # X: (n_samples, n_features) integer hashes
    ranking = rank_features(scores)  # indices, best feature first

``X`` is always treated as **nominal** data: the integer values are category
identifiers, and no arithmetic meaning is attached to their magnitude (the only
exception is the KST, which uses the ordering of the hash domain to build the
empirical CDFs, exactly as described in the manuscript).
"""

from .proposed import (
    kst_score,
    icd_max_score,
    ics_max_score,
    idis_score,
    rank_features,
)
from .baselines import (
    chi_squared_score,
    gain_ratio_score,
    gini_index_score,
    relieff_score,
    lasso_selection,
)
from .registry import FS_METHODS, PROPOSED_METHODS, SOTA_METHODS, get_fs_method

__version__ = "1.0.0"

__all__ = [
    "kst_score",
    "icd_max_score",
    "ics_max_score",
    "idis_score",
    "rank_features",
    "chi_squared_score",
    "gain_ratio_score",
    "gini_index_score",
    "relieff_score",
    "lasso_selection",
    "FS_METHODS",
    "PROPOSED_METHODS",
    "SOTA_METHODS",
    "get_fs_method",
]
