#!/usr/bin/env python3
"""
Experiment 1 - feature selection comparison on a single MinHash sketch.

Reproduces the protocol behind Table 4: for each classifier (SC, RF, XGBoost)
and each FS method (the four proposed ones and the five SOTA baselines), the
top-k features are swept over k = 10, 20, ..., 390, and the best point of the
sweep is compared against the 400-feature baseline. The difference between the
best proposed method and the best SOTA method is tested with a one-sided paired
t-test over the shared cross-validation folds.

Usage
-----
    python scripts/run_experiment1.py --data data/demo/demo_binary.csv \
        --metric auc --classifiers sc rf --out results/experiment1_demo.json

    python scripts/run_experiment1.py --data data/minhash/dna_400.csv \
        --metric accuracy --classifiers sc rf xgb
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
from minhash_fs.stats import paired_ttest_one_sided


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="MinHash CSV (features + label column)")
    p.add_argument("--metric", default="auc", choices=["auc", "accuracy"],
                   help="AUC for binary tasks, accuracy for the multi-class task")
    p.add_argument("--classifiers", nargs="+", default=["sc", "rf"],
                   choices=["sc", "rf", "xgb"])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--step", type=int, default=10, help="step of the top-k grid")
    p.add_argument("--alphas", nargs="+", type=float,
                   default=[0.3, 0.4, 0.5, 0.6, 0.7],
                   help="alpha grid swept for IDIS (Equation 9)")
    p.add_argument("--fs-scope", default="train_fold", choices=["train_fold", "full"],
                   help="where the feature ranking is computed (see README)")
    p.add_argument("--relieff-samples", type=int, default=500,
                   help="instances drawn by ReliefF; lower it to speed the run up")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/experiment1.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_minhash_csv(args.data)
    k_grid = default_k_grid(X.shape[1], step=args.step)
    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} features "
          f"({len(np.unique(y))} classes) from {args.data}")

    report = {
        "dataset": str(args.data),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "metric": args.metric,
        "folds": args.folds,
        "fs_scope": args.fs_scope,
        "seed": args.seed,
        "baselines": {},
        "sweeps": [],
        "significance": {},
    }

    # component scores reused across the alpha grid of IDIS
    icd_full = icd_max_score(X, y)
    ics_full = ics_max_score(X, y)

    for clf in args.classifiers:
        print(f"\n=== classifier: {clf.upper()} ===")

        baseline = cross_validate(X, y, classifier=clf, metric=args.metric,
                                  n_splits=args.folds, random_state=args.seed)
        report["baselines"][clf] = {"mean": baseline.mean, "std": baseline.std,
                                    "fold_scores": baseline.scores,
                                    "n_features": int(X.shape[1])}
        print(f"baseline (all {X.shape[1]} features): {baseline.mean:.4f}")

        sweeps = {}

        # ---- proposed methods -------------------------------------------- #
        for name in ("kst", "icd_max", "ics_max"):
            scorer = PROPOSED_METHODS[name]
            sweep = feature_sweep(X, y, scorer, name, classifier=clf,
                                  metric=args.metric, k_values=k_grid,
                                  n_splits=args.folds, random_state=args.seed,
                                  fs_scope=args.fs_scope)
            sweeps[name] = sweep
            print(f"  {name:<12} best={sweep.best_score:.4f} @ k={sweep.best_k}")

        best_idis = None
        for alpha in args.alphas:
            if args.fs_scope == "full":
                scorer = (lambda Xa, ya, a=alpha:
                          idis_score_from_components(icd_full, ics_full, alpha=a))
            else:
                scorer = (lambda Xa, ya, a=alpha:
                          idis_score_from_components(icd_max_score(Xa, ya),
                                                     ics_max_score(Xa, ya), alpha=a))
            sweep = feature_sweep(X, y, scorer, f"idis(alpha={alpha})",
                                  classifier=clf, metric=args.metric,
                                  k_values=k_grid, n_splits=args.folds,
                                  random_state=args.seed, fs_scope=args.fs_scope)
            if best_idis is None or sweep.best_score > best_idis.best_score:
                best_idis = sweep
        sweeps["idis"] = best_idis
        print(f"  {'idis':<12} best={best_idis.best_score:.4f} @ k={best_idis.best_k} "
              f"[{best_idis.method}]")

        # ---- SOTA baselines ---------------------------------------------- #
        for name, scorer in SOTA_METHODS.items():
            if name == "relieff":
                scorer = (lambda Xa, ya:
                          SOTA_METHODS["relieff"](Xa, ya,
                                                  n_samples_draw=args.relieff_samples,
                                                  random_state=args.seed))
            sweep = feature_sweep(X, y, scorer, name, classifier=clf,
                                  metric=args.metric, k_values=k_grid,
                                  n_splits=args.folds, random_state=args.seed,
                                  fs_scope=args.fs_scope)
            sweeps[name] = sweep
            print(f"  {name:<12} best={sweep.best_score:.4f} @ k={sweep.best_k}")

        # ---- comparison of the two best methods --------------------------- #
        best_proposed = max((sweeps[n] for n in ("kst", "icd_max", "ics_max", "idis")),
                            key=lambda s: s.best_score)
        best_sota = max((sweeps[n] for n in SOTA_METHODS), key=lambda s: s.best_score)
        test = paired_ttest_one_sided(best_proposed.fold_scores[best_proposed.best_k],
                                      best_sota.fold_scores[best_sota.best_k])

        report["significance"][clf] = {
            "best_proposed": best_proposed.method,
            "best_proposed_score": best_proposed.best_score,
            "best_proposed_k": best_proposed.best_k,
            "improvement_over_baseline_pct":
                100.0 * (best_proposed.best_score - baseline.mean) / baseline.mean,
            "best_sota": best_sota.method,
            "best_sota_score": best_sota.best_score,
            "best_sota_k": best_sota.best_k,
            "sota_improvement_over_baseline_pct":
                100.0 * (best_sota.best_score - baseline.mean) / baseline.mean,
            "t_statistic": test.statistic,
            "p_value": test.p_value,
            "effect_size_cohens_d": test.effect_size,
            "significant_at_0.05": test.significant,
        }
        report["sweeps"].extend(s.to_dict() for s in sweeps.values())

        print(f"  -> best proposed: {best_proposed.method} ({best_proposed.best_score:.4f}"
              f" @ k={best_proposed.best_k})")
        print(f"  -> best SOTA    : {best_sota.method} ({best_sota.best_score:.4f}"
              f" @ k={best_sota.best_k})")
        print(f"  -> paired t-test: {test}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
