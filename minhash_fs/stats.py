"""
Statistical testing used in Tables 4 and 5.

The significance of the difference between a proposed FS method and the best
performing SOTA method is assessed with a **one-sided paired t-test** over the
per-fold scores of the same 10-fold cross-validation split, i.e. the two methods
are always compared on identical folds.

The paired design is what makes the test appropriate here: the folds are shared,
so the fold-to-fold variability is removed from the comparison. The test is
one-sided because the hypothesis under evaluation is directional (the proposed
method is *not worse* than the baseline).

Caveat, reported alongside the results: repeated cross-validation folds are not
independent, so the p-values of a paired t-test over CV folds are known to be
optimistic. They are reported as a comparative indication rather than as an
exact inferential guarantee, and the effect sizes are reported next to them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["PairedTestResult", "paired_ttest_one_sided", "cohens_d_paired"]


@dataclass
class PairedTestResult:
    statistic: float
    p_value: float
    mean_difference: float
    effect_size: float
    n_folds: int
    alternative: str = "greater"

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def __str__(self) -> str:      # pragma: no cover - cosmetic only
        verdict = "significant" if self.significant else "not significant"
        return (f"t={self.statistic:.3f}, p={self.p_value:.4g} ({verdict}), "
                f"mean diff={self.mean_difference:+.4f}, d={self.effect_size:.3f}, "
                f"n={self.n_folds}")


def cohens_d_paired(a, b) -> float:
    """Cohen's d for paired samples (mean difference / SD of the differences)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def paired_ttest_one_sided(proposed, baseline,
                           alternative: str = "greater") -> PairedTestResult:
    """One-sided paired t-test between two vectors of per-fold scores.

    Parameters
    ----------
    proposed, baseline : sequence of float
        Per-fold scores of the two methods, in the same fold order.
    alternative : {"greater", "less"}
        ``"greater"`` tests whether ``proposed`` exceeds ``baseline``.
    """
    proposed = np.asarray(proposed, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if proposed.shape != baseline.shape:
        raise ValueError("both score vectors must have the same length")
    if proposed.size < 2:
        raise ValueError("at least two folds are required")

    differences = proposed - baseline
    if np.allclose(differences, 0.0):
        # the two methods scored identically on every fold: the t statistic is
        # undefined, and the correct conclusion is "no detectable difference"
        return PairedTestResult(statistic=0.0, p_value=1.0, mean_difference=0.0,
                                effect_size=0.0, n_folds=int(proposed.size),
                                alternative=alternative)

    result = stats.ttest_rel(proposed, baseline, alternative=alternative)
    return PairedTestResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        mean_difference=float(np.mean(proposed - baseline)),
        effect_size=cohens_d_paired(proposed, baseline),
        n_folds=int(proposed.size),
        alternative=alternative,
    )
