#!/usr/bin/env python3
"""
Experiment 2 - combining two MinHash sketches, then selecting features.

Two sketches of 200 features each are generated for every item, using the same
shingle size but different overlaps (Table 3). They are concatenated into a
single 400-feature vector, and the FS methods of Experiment 1 are applied to the
combined representation. The script reports, for each classifier:

  * the performance of each individual 200-feature sketch,
  * the performance of the 400-feature concatenation (the aggregation gain),
  * the best top-k result of every FS method on the concatenation,
  * a one-sided paired t-test between the best proposed and the best SOTA method.

This reproduces the protocol behind Table 5.

Usage
-----
    python scripts/run_experiment2.py \
        --sketch-a data/minhash/pdf_overlap64_200.csv \
        --sketch-b data/minhash/pdf_overlap128_200.csv \
        --metric auc --classifiers sc rf xgb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minhash_fs.datasets import load_minhash_csv
from minhash_fs.evaluation import cross_validate, feature_sweep, default_k_grid
from minhash_fs.proposed import icd_max_score, ics_max_score, idis_score_from_components
from minhash_fs.registry import PROPOSED_METHODS, SOTA_METHODS
from minhash_fs.sketch import combine_sketches
from minhash_fs.stats import paired_ttest_one_sided


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sketch-a", required=True, help="first sketch CSV (200 features)")
    p.add_argument("--sketch-b", required=True, help="second sketch CSV (200 features)")
    p.add_argument("--metric", default="auc", choices=["auc", "accuracy"])
    p.add_argument("--classifiers", nargs="+", default=["sc", "rf"],
                   choices=["sc", "rf", "xgb"])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--step", type=int, default=10)
    p.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7])
    p.add_argument("--fs-scope", default="train_fold", choices=["train_fold", "full"])
    p.add_argument("--relieff-samples", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/experiment2.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    Xa, ya = load_minhash_csv(args.sketch_a)
    Xb, yb = load_minhash_csv(args.sketch_b)
    if not np.array_equal(ya, yb):
        raise ValueError("the two sketch files must describe the same items, "
                         "in the same order, with identical labels")

    X = combine_sketches(Xa, Xb)
    y = ya
    k_grid = default_k_grid(X.shape[1], step=args.step)
    print(f"Sketch A: {Xa.shape}, sketch B: {Xb.shape}, combined: {X.shape}")

    report = {
        "sketch_a": str(args.sketch_a), "sketch_b": str(args.sketch_b),
        "n_samples": int(X.shape[0]), "n_features": int(X.shape[1]),
        "metric": args.metric, "folds": args.folds, "fs_scope": args.fs_scope,
        "seed": args.seed, "aggregation": {}, "sweeps": [], "significance": {},
    }

    icd_full = icd_max_score(X, y)
    ics_full = ics_max_score(X, y)

    for clf in args.classifiers:
        print(f"\n=== classifier: {clf.upper()} ===")

        single_a = cross_validate(Xa, ya, classifier=clf, metric=args.metric,
                                  n_splits=args.folds, random_state=args.seed)
        single_b = cross_validate(Xb, yb, classifier=clf, metric=args.metric,
                                  n_splits=args.folds, random_state=args.seed)
        combined = cross_validate(X, y, classifier=clf, metric=args.metric,
                                  n_splits=args.folds, random_state=args.seed)
        best_single = max(single_a.mean, single_b.mean)
        report["aggregation"][clf] = {
            "sketch_a": single_a.mean, "sketch_b": single_b.mean,
            "combined": combined.mean,
            "gain_over_best_single_pct":
                100.0 * (combined.mean - best_single) / best_single,
            "combined_fold_scores": combined.scores,
        }
        print(f"sketch A: {single_a.mean:.4f} | sketch B: {single_b.mean:.4f} "
              f"| combined: {combined.mean:.4f}")

        sweeps = {}
        for name in ("kst", "icd_max", "ics_max"):
            sweep = feature_sweep(X, y, PROPOSED_METHODS[name], name, classifier=clf,
                                  metric=args.metric, k_values=k_grid,
                                  n_splits=args.folds, random_state=args.seed,
                                  fs_scope=args.fs_scope)
            sweeps[name] = sweep
            print(f"  {name:<12} best={sweep.best_score:.4f} @ k={sweep.best_k}")

        best_idis = None
        for alpha in args.alphas:
            if args.fs_scope == "full":
                scorer = (lambda Xa_, ya_, a=alpha:
                          idis_score_from_components(icd_full, ics_full, alpha=a))
            else:
                scorer = (lambda Xa_, ya_, a=alpha:
                          idis_score_from_components(icd_max_score(Xa_, ya_),
                                                     ics_max_score(Xa_, ya_), alpha=a))
            sweep = feature_sweep(X, y, scorer, f"idis(alpha={alpha})", classifier=clf,
                                  metric=args.metric, k_values=k_grid,
                                  n_splits=args.folds, random_state=args.seed,
                                  fs_scope=args.fs_scope)
            if best_idis is None or sweep.best_score > best_idis.best_score:
                best_idis = sweep
        sweeps["idis"] = best_idis
        print(f"  {'idis':<12} best={best_idis.best_score:.4f} @ k={best_idis.best_k} "
              f"[{best_idis.method}]")

        for name, scorer in SOTA_METHODS.items():
            if name == "relieff":
                scorer = (lambda Xa_, ya_:
                          SOTA_METHODS["relieff"](Xa_, ya_,
                                                  n_samples_draw=args.relieff_samples,
                                                  random_state=args.seed))
            sweep = feature_sweep(X, y, scorer, name, classifier=clf,
                                  metric=args.metric, k_values=k_grid,
                                  n_splits=args.folds, random_state=args.seed,
                                  fs_scope=args.fs_scope)
            sweeps[name] = sweep
            print(f"  {name:<12} best={sweep.best_score:.4f} @ k={sweep.best_k}")

        best_proposed = max((sweeps[n] for n in ("kst", "icd_max", "ics_max", "idis")),
                            key=lambda s: s.best_score)
        best_sota = max((sweeps[n] for n in SOTA_METHODS), key=lambda s: s.best_score)
        test = paired_ttest_one_sided(best_proposed.fold_scores[best_proposed.best_k],
                                      best_sota.fold_scores[best_sota.best_k])

        report["significance"][clf] = {
            "best_proposed": best_proposed.method,
            "best_proposed_score": best_proposed.best_score,
            "best_proposed_k": best_proposed.best_k,
            "best_sota": best_sota.method,
            "best_sota_score": best_sota.best_score,
            "best_sota_k": best_sota.best_k,
            "p_value": test.p_value,
            "effect_size_cohens_d": test.effect_size,
            "significant_at_0.05": test.significant,
        }
        report["sweeps"].extend(s.to_dict() for s in sweeps.values())
        print(f"  -> paired t-test ({best_proposed.method} vs {best_sota.method}): {test}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
