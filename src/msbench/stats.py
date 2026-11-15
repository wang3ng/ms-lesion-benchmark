"""Statistics for comparing models across patients.

All comparisons are paired by patient: the same test cases scored by two
models. Use these on per-case metric values from ``msbench.metrics``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy import stats as sps


def _clean_pairs(a: Sequence[float], b: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired inputs must have the same length")
    keep = ~(np.isnan(a) | np.isnan(b))
    return a[keep], b[keep]


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test of a vs b (paired by case).

    Cases where either value is NaN are dropped. Returns the median
    difference (a - b), the statistic, p-value and number of pairs used.
    """
    a, b = _clean_pairs(a, b)
    diff = a - b
    if len(diff) == 0 or np.allclose(diff, 0):
        return {"n": int(len(diff)), "median_diff": 0.0, "statistic": float("nan"), "p": 1.0}
    res = sps.wilcoxon(a, b, alternative="two-sided")
    return {
        "n": int(len(diff)),
        "median_diff": float(np.median(diff)),
        "statistic": float(res.statistic),
        "p": float(res.pvalue),
    }


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 20261011,
) -> dict:
    """Percentile bootstrap CI over cases (resampling patients)."""
    x = np.asarray(values, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0, "estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = np.array([stat(x[i]) for i in idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"n": int(len(x)), "estimate": float(stat(x)), "low": float(lo), "high": float(hi)}


def paired_bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    stat: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    level: float = 0.95,
    seed: int = 20261011,
) -> dict:
    """Bootstrap CI for stat(a) - stat(b), resampling the same cases for both."""
    a, b = _clean_pairs(a, b)
    if len(a) == 0:
        return {"n": 0, "estimate": float("nan"), "low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    boots = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {
        "n": int(len(a)),
        "estimate": float(stat(a) - stat(b)),
        "low": float(lo),
        "high": float(hi),
    }


def bland_altman(auto: Sequence[float], manual: Sequence[float]) -> dict:
    """Bland-Altman agreement between automatic and manual measurements.

    Returns bias (mean of auto - manual), 95% limits of agreement, MAE, and
    the per-case means/differences for plotting.
    """
    a, m = _clean_pairs(auto, manual)
    diff = a - m
    if len(diff) == 0:
        nan = float("nan")
        return {"n": 0, "bias": nan, "loa_low": nan, "loa_high": nan, "mae": nan,
                "means": [], "diffs": []}
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    bias = float(np.mean(diff))
    return {
        "n": int(len(diff)),
        "bias": bias,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "mae": float(np.mean(np.abs(diff))),
        "means": ((a + m) / 2).tolist(),
        "diffs": diff.tolist(),
    }
