"""Bootstrap confidence intervals and paired comparisons on a fixed test set.

Resampling is stratified by label by default, so every replicate keeps the (very low) fraud
prevalence of the test set and threshold-based metrics stay well defined.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Interval:
    """Point estimate with a percentile bootstrap interval."""

    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _bootstrap_indices(strata: np.ndarray | None, n: int, n_boot: int, rng: np.random.Generator):
    if strata is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, size=n)
        return
    members = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    for _ in range(n_boot):
        yield np.concatenate([rng.choice(m, size=m.size, replace=True) for m in members])


def bootstrap_ci(
    metric: Callable[..., float],
    *arrays: ArrayLike,
    n_boot: int = 1000,
    alpha: float = 0.05,
    strata: ArrayLike | None = None,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI of ``metric(*arrays)``; all arrays are resampled together.

    Pass ``strata=y_true`` (the usual case) to resample fraud and non-fraud rows separately.
    """
    data = [np.asarray(a) for a in arrays]
    n = len(data[0])
    if any(len(a) != n for a in data):
        raise ValueError("All arrays must have the same length.")
    rng = np.random.default_rng(seed)
    strata_arr = None if strata is None else np.asarray(strata)
    estimate = float(metric(*data))
    replicates = np.array(
        [metric(*(a[idx] for a in data)) for idx in _bootstrap_indices(strata_arr, n, n_boot, rng)]
    )
    low, high = np.nanquantile(replicates, [alpha / 2, 1 - alpha / 2])
    return Interval(estimate, float(low), float(high))


def paired_bootstrap_diff(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true: ArrayLike,
    score_a: ArrayLike,
    score_b: ArrayLike,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[Interval, float]:
    """CI of ``metric(A) - metric(B)`` on shared resamples, plus a two-sided p-value.

    The p-value is the bootstrap share of differences on the other side of zero, doubled.
    """
    y = np.asarray(y_true)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    estimate = metric(y, a) - metric(y, b)
    diffs = np.array(
        [
            metric(y[i], a[i]) - metric(y[i], b[i])
            for i in _bootstrap_indices(y, len(y), n_boot, rng)
        ]
    )
    low, high = np.nanquantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_value = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
    return Interval(float(estimate), float(low), float(high)), float(p_value)
