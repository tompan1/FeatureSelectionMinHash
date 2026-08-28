# Feature Selection Strategies for MinHash-Based Classifiers

Reference implementation accompanying:

> T. Panker and N. Nissim, "Feature Selection Strategies for Improving
> MinHash-Based Classifiers' Performance and Generalization."
> Malware Lab, Cyber Security Research Center, Ben-Gurion University of the Negev.

This repository contains the four feature selection (FS) methods proposed in the
paper, the five baselines they are compared against, the MinHash sketch
generation pipeline, and the complete evaluation protocol used in Experiments 1
and 2.

MinHash features are **nominal**: each value is a category identifier produced by
an independent hash function, and its numerical magnitude carries no meaning.
The methods below are designed around that constraint, which is what makes
conventional FS techniques a poor fit for MinHash representations.

---

## The four proposed methods

| Method | Idea | Paper | Complexity (binary / multi-class) |
|--------|------|-------|-----------------------------------|
| **KST** | Rank features by the Kolmogorov–Smirnov distance between the class-conditional distributions of their hash values | Algorithm 1 | `O(M·N log N)` / `O(M·C·N log N)` |
| **ICD-Max** | Maximise inter-class dissimilarity: prefer features whose classes use disjoint sets of hashes (low Jaccard similarity between classes) | Algorithm 2 | `O(M·N)` / `O(M·C·N)` |
| **ICS-Max** | Maximise intra-class similarity: reward hash values that repeat inside one class, ignore singletons | Algorithm 3 | `O(M·N)` / `O(M·N)` |
| **IDIS** | Weighted combination, `α·ICD-Max + (1−α)·ICS-Max` | Equation 9 | `O(M·C·N)` |

`N` = samples, `M` = features, `C` = classes.

Baselines: chi-squared, gain ratio, Gini index, ReliefF, and L1 regularisation
(Lasso). The first four operate directly on the nominal hash values, so every
method is compared on exactly the same representation.

---

## Installation

```bash
git clone <repository-url>
cd minhash-fs
pip install -r requirements.txt
python -m pytest tests -q          # 26 tests, ~2 s
```

Python 3.9+ is required. `xgboost` is optional and only needed for the XGB
classifier; everything else runs on NumPy, SciPy and scikit-learn.

---

## Quick start

Every FS method shares one interface:

```python
import numpy as np
from minhash_fs import kst_score, icd_max_score, ics_max_score, idis_score, rank_features

# X: (n_samples, n_features) integer MinHash values; y: labels
scores  = idis_score(X, y, alpha=0.5)   # higher is better
ranking = rank_features(scores)         # feature indices, best first
X_top50 = X[:, np.sort(ranking[:50])]   # the 50 most useful features
```

Run the whole pipeline on the bundled synthetic data, without needing any of the
restricted corpora:

```bash
python scripts/make_demo_dataset.py --out data/demo/demo_binary.csv
python scripts/run_experiment1.py --data data/demo/demo_binary.csv \
    --metric auc --classifiers sc rf --folds 5 --step 8
```

---

## Repository layout

```
minhash_fs/
├── proposed.py      KST, ICD-Max, ICS-Max, IDIS  (Algorithms 1–3, Equation 9)
├── baselines.py     chi-squared, gain ratio, Gini index, ReliefF, Lasso
├── rabin.py         Rabin fingerprinting over GF(2), irreducible polynomials
├── shingles.py      byte / word / character / 2-D pixel shingle extraction
├── sketch.py        MinHash sketch construction, partition MinHash, aggregation
├── classifiers.py   Similarity Classifier (SC), Random Forest, XGBoost
├── evaluation.py    10-fold CV, top-k sweep, AUC / accuracy
├── stats.py         one-sided paired t-test, Cohen's d
├── datasets.py      CSV / NPZ I/O, the toy example of Figure 1
└── registry.py      name → scorer lookup used by the scripts

scripts/
├── build_sketches.py        raw data  → MinHash CSV
├── make_demo_dataset.py     synthetic MinHash CSV for smoke tests
├── run_experiment1.py       Experiment 1 (Table 4)
├── run_experiment2.py       Experiment 2 (Table 5)
└── toy_example_ranking.py   FS behaviour on the toy dataset of Section 3.D

configs/datasets.yaml        shingle lengths and overlaps (Tables 2 and 3)
data/                        data availability statement, hash lists, demo data
tests/                       unit tests
```

---

## Reproducing the experiments

### Step 1 — obtain the raw data

See [`data/README.md`](data/README.md). Four datasets are public third-party
corpora; the malware samples of the dynamic-analysis dataset are identified by
SHA-256 rather than redistributed.

### Step 2 — build the sketches

```bash
# Experiment 1: one 400-feature sketch per item
python scripts/build_sketches.py --dataset dna --input raw/dna \
    --n-functions 400 --out data/minhash/dna_400.csv

# Experiment 2: two 200-feature sketches, same shingle size, different overlap
python scripts/build_sketches.py --dataset dna --input raw/dna \
    --variant primary   --n-functions 200 --out data/minhash/dna_a_200.csv
python scripts/build_sketches.py --dataset dna --input raw/dna \
    --variant secondary --n-functions 200 --out data/minhash/dna_b_200.csv
```

400 hash functions keep the Jaccard estimation error at roughly `1/√400 = 0.05`,
which is the configuration used throughout the paper.

### Step 3 — run the experiments

```bash
# Experiment 1 — FS comparison on a single sketch (Table 4)
python scripts/run_experiment1.py --data data/minhash/dna_400.csv \
    --metric accuracy --classifiers sc rf xgb --out results/experiment1_dna.json

# Experiment 2 — sketch combination + FS (Table 5)
python scripts/run_experiment2.py \
    --sketch-a data/minhash/dna_a_200.csv \
    --sketch-b data/minhash/dna_b_200.csv \
    --metric accuracy --classifiers sc rf xgb --out results/experiment2_dna.json
```

Use `--metric auc` for the four binary tasks and `--metric accuracy` for the
multi-class DNA task, as in the paper.

Each run writes a JSON report containing the per-fold scores of every
(method, classifier, k) configuration, the best point of each sweep, the number
of features at which it was reached, and the paired t-test against the best
baseline. Because the per-fold scores are stored, every number in the paper can
be recomputed from the JSON without rerunning the experiments.

---

## Evaluation protocol

* Stratified **10-fold cross-validation**; identical folds for every method, so
  the paired t-test compares like with like.
* Classifier hyper-parameters are **fixed and never tuned** (SC: average Jaccard
  similarity to each class; RF: 100 unpruned trees; XGBoost: 100 rounds, fixed
  learning rate, bounded depth), so that any difference in performance is
  attributable to the feature selection method.
* Top-k sweep over `k = 10, 20, …, 390`, with all 400 features as the baseline.
* AUC for binary tasks, accuracy for the multi-class task.
* Significance: one-sided paired t-test over the shared folds, reported together
  with Cohen's *d*.

### Where the feature ranking is computed

`--fs-scope` controls whether the ranking is recomputed inside each training
fold or once on the whole dataset:

| value | behaviour | when to use |
|-------|-----------|-------------|
| `train_fold` (default) | the ranking is recomputed on each training fold, so the held-out fold never influences selection | recommended for any new comparison; leakage-free |
| `full` | the ranking is computed once on the whole dataset and reused in every fold | faster; provided for comparability with published protocols that follow this convention |

The scope actually used is recorded in every results file, so no reported number
is ambiguous.

---

## Notes and caveats

* **Feature selection does not preserve Jaccard similarity.** Discarding hash
  positions breaks the probabilistic guarantee that makes MinHash an unbiased
  similarity estimator. This is a deliberate trade-off: the objective here is
  classification performance, not similarity preservation. Sketches intended for
  similarity search should not be pruned with these methods.
* **The KST uses the ordering of the hash domain**, not the arithmetic of the
  hash values. No addition, averaging or scaling is ever applied to a hash.
* **Paired t-tests over cross-validation folds are known to be optimistic**,
  because the folds share training data and are therefore not independent. The
  p-values are reported as a comparative indication, alongside effect sizes,
  rather than as exact inferential guarantees.
* **Lasso requires a numeric reading of nominal values**, which contradicts the
  nature of MinHash features. It is included precisely because it is a standard
  choice for n-gram features, and the paper reports that it transfers poorly.
* **ICS-Max multi-class rule.** The prose of Section 6.C credits a hash value to
  the class in which it is most frequent (`multiclass_rule="argmax"`, the
  default); the pseudo-code of Algorithm 3 additionally requires that count to
  exceed the total in all other classes (`multiclass_rule="vs_rest"`). The two
  rules coincide for binary tasks. Both are implemented.

---

## Citation

```bibtex
@article{panker_minhash_fs,
  title   = {Feature Selection Strategies for Improving MinHash-Based
             Classifiers' Performance and Generalization},
  author  = {Panker, Tomer and Nissim, Nir},
  note    = {Malware Lab, Cyber Security Research Center,
             Ben-Gurion University of the Negev},
  year    = {2026}
}
```

## License

Released under the MIT License; see [`LICENSE`](LICENSE).
