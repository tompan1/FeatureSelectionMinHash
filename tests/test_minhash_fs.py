"""
Unit tests for the feature selection package.

Run with:   python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minhash_fs.baselines import (chi_squared_score, gain_ratio_score,
                                  gini_index_score, relieff_score, lasso_selection)
from minhash_fs.classifiers import SimilarityClassifier, minhash_similarity
from minhash_fs.datasets import (load_minhash_csv, save_minhash_csv, toy_example)
from minhash_fs.evaluation import cross_validate, feature_sweep, default_k_grid
from minhash_fs.proposed import (icd_max_score, ics_max_score, idis_score,
                                 ks_statistic, kst_score, rank_features)
from minhash_fs.rabin import (RabinFingerprint, is_irreducible,
                              make_fingerprint_family, random_irreducible_polynomial)
from minhash_fs.shingles import byte_shingles, character_shingles, word_shingles
from minhash_fs.sketch import combine_sketches, minhash_sketch
from minhash_fs.stats import paired_ttest_one_sided


# --------------------------------------------------------------------------- #
# Separable / non-separable fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def separable():
    """Feature 0 separates the classes perfectly; feature 1 carries no signal."""
    X = np.array([[10, 99], [10, 99], [11, 99], [11, 99]], dtype=np.int64)
    y = np.array([0, 0, 1, 1])
    return X, y


@pytest.fixture
def multiclass():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 20)
    good = np.repeat([100, 200, 300], 20)                  # one value per class
    noise = rng.integers(0, 5, size=y.size) + 900          # identical everywhere
    return np.column_stack([good, noise]).astype(np.int64), y


# --------------------------------------------------------------------------- #
# Proposed methods
# --------------------------------------------------------------------------- #
def test_kst_prefers_separable_feature(separable):
    X, y = separable
    scores = kst_score(X, y)
    assert scores[0] > scores[1]
    assert rank_features(scores)[0] == 0


def test_ks_statistic_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=50), rng.normal(loc=1.0, size=70)
    from scipy.stats import ks_2samp
    assert ks_statistic(a, b) == pytest.approx(ks_2samp(a, b).statistic)


def test_icd_max_bounds_and_ordering(separable):
    X, y = separable
    scores = icd_max_score(X, y)
    assert scores[0] == pytest.approx(1.0)   # disjoint hash sets -> 1 - J = 1
    assert scores[1] == pytest.approx(0.0)   # identical hash sets -> 1 - J = 0
    assert np.all((scores >= 0) & (scores <= 1))


def test_ics_max_ignores_singletons():
    # feature 0: every value unique -> nothing counted
    # feature 1: one value repeated inside class 0 only
    X = np.array([[1, 50], [2, 50], [3, 60], [4, 61]], dtype=np.int64)
    y = np.array([0, 0, 1, 1])
    scores = ics_max_score(X, y)
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(0.5)   # 2 beneficial occurrences / 4 samples


def test_ics_max_multiclass_rules_agree_on_binary(separable):
    X, y = separable
    assert np.allclose(ics_max_score(X, y, multiclass_rule="argmax"),
                       ics_max_score(X, y, multiclass_rule="vs_rest"))


def test_idis_is_a_convex_combination(separable):
    X, y = separable
    icd, ics = icd_max_score(X, y), ics_max_score(X, y)
    for alpha in (0.0, 0.25, 0.5, 1.0):
        assert np.allclose(idis_score(X, y, alpha=alpha), alpha * icd + (1 - alpha) * ics)
    with pytest.raises(ValueError):
        idis_score(X, y, alpha=1.5)


def test_all_methods_handle_multiclass(multiclass):
    X, y = multiclass
    for scorer in (kst_score, icd_max_score, ics_max_score, idis_score,
                   chi_squared_score, gain_ratio_score, gini_index_score):
        scores = np.asarray(scorer(X, y))
        assert scores.shape == (X.shape[1],)
        assert np.all(np.isfinite(scores))
        assert scores[0] >= scores[1], f"{scorer.__name__} ranked the noise feature first"


def test_rank_features_is_deterministic_under_ties():
    scores = np.array([0.5, 0.5, 0.9, 0.1])
    order = rank_features(scores)
    assert order[0] == 2 and order[-1] == 3
    assert list(order[1:3]) == [0, 1]        # ties broken by index


# --------------------------------------------------------------------------- #
# Section 3.D claims on the toy dataset of Figure 1
# --------------------------------------------------------------------------- #
def test_toy_example_extreme_cases():
    X, y, names = toy_example()
    # f0 is constant -> no discriminative power under any proposed method
    for scorer in (kst_score, icd_max_score, ics_max_score):
        scores = scorer(X, y)
        assert scores[0] == pytest.approx(min(scores)), f"{scorer.__name__} on f0"
    # f1 separates the classes perfectly -> top-ranked by ICD-Max and KST
    assert rank_features(icd_max_score(X, y))[0] == 1
    assert rank_features(kst_score(X, y))[0] == 1
    # f2 has a unique value per sample -> ICS-Max assigns it a zero score
    assert ics_max_score(X, y)[2] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def test_baselines_return_finite_scores(separable):
    X, y = separable
    for scorer in (chi_squared_score, gain_ratio_score, gini_index_score):
        scores = scorer(X, y)
        assert scores.shape == (2,)
        assert np.all(np.isfinite(scores))
        assert scores[0] > scores[1]


def test_relieff_prefers_separable_feature():
    rng = np.random.default_rng(3)
    y = np.repeat([0, 1], 25)
    good = np.repeat([7, 8], 25)
    noise = rng.integers(0, 10, size=50)
    X = np.column_stack([good, noise]).astype(np.int64)
    scores = relieff_score(X, y, n_neighbors=5, n_samples_draw=None)
    assert scores[0] > scores[1]


def test_lasso_returns_scores_and_indices():
    rng = np.random.default_rng(4)
    y = np.repeat([0, 1], 30)
    X = np.column_stack([np.repeat([10, 90], 30),
                         rng.integers(0, 100, size=60)]).astype(np.int64)
    scores, selected = lasso_selection(X, y)
    assert scores.shape == (2,)
    assert set(selected).issubset({0, 1})


# --------------------------------------------------------------------------- #
# Rabin fingerprinting and sketching
# --------------------------------------------------------------------------- #
def test_irreducibility_test_on_known_polynomials():
    assert is_irreducible(0b111)          # x^2 + x + 1 is irreducible
    assert not is_irreducible(0b101)      # x^2 + 1 = (x + 1)^2
    assert is_irreducible(0b1011)         # x^3 + x + 1


def test_random_polynomial_has_requested_degree():
    rng = np.random.default_rng(5)
    for degree in (8, 16, 31):
        p = random_irreducible_polynomial(degree, rng)
        assert p.bit_length() - 1 == degree
        assert is_irreducible(p)


def test_fingerprint_is_deterministic_and_bounded():
    rng = np.random.default_rng(6)
    f = RabinFingerprint(random_irreducible_polynomial(31, rng))
    assert f(b"abcdef") == f(b"abcdef")
    assert f(b"abcdef") != f(b"abcdeg")
    assert 0 <= f(b"abcdef") < 2 ** 31


def test_sketch_of_identical_items_is_identical():
    family = make_fingerprint_family(8, degree=31, random_state=1)
    data = b"the quick brown fox jumps over the lazy dog" * 4
    s1 = minhash_sketch(byte_shingles(data, 8, 4), family)
    s2 = minhash_sketch(byte_shingles(data, 8, 4), family)
    assert np.array_equal(s1, s2)


def test_sketch_similarity_tracks_jaccard():
    """More overlapping content -> higher fraction of matching sketch entries."""
    family = make_fingerprint_family(64, degree=31, random_state=2)
    base = bytes(range(256)) * 4
    near = base[:900] + bytes(range(50)) * 2
    far = bytes(reversed(bytes(range(200)))) * 5

    s_base = minhash_sketch(byte_shingles(base, 8, 4), family)
    s_near = minhash_sketch(byte_shingles(near, 8, 4), family)
    s_far = minhash_sketch(byte_shingles(far, 8, 4), family)

    sim_near = float(np.mean(s_base == s_near))
    sim_far = float(np.mean(s_base == s_far))
    assert sim_near > sim_far


def test_shingle_extractors_respect_stride():
    assert len(list(byte_shingles(bytes(range(20)), 8, 4))) == 4
    assert len(list(word_shingles("a b c d e f", 3, 2))) == 4
    assert len(list(character_shingles("ACGTACGTAC", 4, 2))) == 4
    with pytest.raises(ValueError):
        list(byte_shingles(b"abc", 2, 2))       # overlap must be < length


def test_combine_sketches_shapes():
    a = np.zeros((5, 3), dtype=np.int64)
    b = np.ones((5, 4), dtype=np.int64)
    assert combine_sketches(a, b).shape == (5, 7)
    with pytest.raises(ValueError):
        combine_sketches(a, np.ones((4, 4), dtype=np.int64))


# --------------------------------------------------------------------------- #
# Classifier and evaluation
# --------------------------------------------------------------------------- #
def test_similarity_classifier_separates_clean_classes():
    X = np.array([[1, 1, 1], [1, 1, 2], [9, 9, 9], [9, 9, 8]], dtype=np.int64)
    y = np.array([0, 0, 1, 1])
    clf = SimilarityClassifier().fit(X, y)
    assert list(clf.predict(X)) == list(y)
    proba = clf.predict_proba(X)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_minhash_similarity_is_the_matching_fraction():
    A = np.array([[1, 2, 3]])
    B = np.array([[1, 2, 3], [1, 0, 0]])
    sim = minhash_similarity(A, B)
    assert sim[0, 0] == pytest.approx(1.0)
    assert sim[0, 1] == pytest.approx(1 / 3)


def test_cross_validate_and_sweep_run_end_to_end():
    rng = np.random.default_rng(8)
    y = np.repeat([0, 1], 40)
    signal = np.repeat([[11, 12], [21, 22]], 40, axis=0)
    noise = rng.integers(0, 6, size=(80, 4)) + 500
    X = np.hstack([signal, noise]).astype(np.int64)

    baseline = cross_validate(X, y, classifier="sc", metric="auc", n_splits=5)
    assert 0.0 <= baseline.mean <= 1.0

    sweep = feature_sweep(X, y, icd_max_score, "icd_max", classifier="sc",
                          metric="auc", k_values=[1, 2, 4], n_splits=5,
                          fs_scope="train_fold")
    assert sweep.best_k in (1, 2, 4)
    assert len(sweep.fold_scores[sweep.best_k]) == 5
    assert sweep.best_score >= baseline.mean - 1e-9


def test_both_fs_scopes_are_supported():
    rng = np.random.default_rng(9)
    y = np.repeat([0, 1], 30)
    X = np.hstack([np.repeat([[1], [2]], 30, axis=0),
                   rng.integers(0, 5, size=(60, 3))]).astype(np.int64)
    for scope in ("train_fold", "full"):
        sweep = feature_sweep(X, y, kst_score, "kst", classifier="sc", metric="auc",
                              k_values=[1, 2], n_splits=5, fs_scope=scope)
        assert sweep.fs_scope == scope


def test_default_k_grid_matches_the_paper():
    grid = default_k_grid(400, step=10)
    assert grid[0] == 10 and grid[-1] == 390 and len(grid) == 39


# --------------------------------------------------------------------------- #
# Statistics and I/O
# --------------------------------------------------------------------------- #
def test_paired_ttest_detects_a_consistent_gain():
    proposed = np.array([0.80, 0.82, 0.81, 0.83, 0.79, 0.84, 0.82, 0.81, 0.80, 0.83])
    baseline = proposed - 0.02
    result = paired_ttest_one_sided(proposed, baseline)
    assert result.significant
    assert result.mean_difference == pytest.approx(0.02)
    assert paired_ttest_one_sided(baseline, proposed).p_value > 0.5


def test_csv_round_trip(tmp_path):
    X = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([0, 1])
    path = save_minhash_csv(tmp_path / "sketch.csv", X, y)
    X2, y2 = load_minhash_csv(path)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
