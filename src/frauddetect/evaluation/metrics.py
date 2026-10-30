"""Performance metrics at the fraud-detection operating point.

The BAF benchmark fixes the decision threshold where the false-positive rate (FPR) stays
*strictly below* 5% and reports the true-positive rate (TPR, recall) there. A score at or
above the threshold counts as flagged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def _as_arrays(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(y_score, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and y_score must be 1-D arrays of equal length.")
    if np.isnan(s).any():
        raise ValueError("y_score contains NaN.")
    return y, s


def threshold_at_fpr(y_true: ArrayLike, y_score: ArrayLike, target_fpr: float = 0.05) -> float:
    """Lowest threshold whose FPR is strictly below ``target_fpr``.

    Among all thresholds meeting the FPR constraint, the lowest one flags the most
    applications and therefore maximises TPR. Only negatives are required, so the
    function also works on a group without any fraud case.
    """
    if not 0 < target_fpr <= 1:
        raise ValueError("target_fpr must lie in (0, 1].")
    y, s = _as_arrays(y_true, y_score)
    negatives = np.sort(s[~y])[::-1]  # descending
    n_neg = negatives.size
    if n_neg == 0:
        raise ValueError("At least one negative example is needed to control the FPR.")

    # Largest number of negatives that may be flagged: k / n_neg < target_fpr.
    k = int(np.ceil(target_fpr * n_neg)) - 1
    if k >= n_neg:
        return float(s.min())  # every negative may be flagged
    # The (k+1)-th highest negative score must stay unflagged, so the threshold must lie
    # strictly above it; the lowest observed score above it gives the same predictions
    # as any threshold in that gap and the highest TPR.
    boundary = negatives[k]
    above = s[s > boundary]
    return float(above.min()) if above.size else float(np.nextafter(boundary, np.inf))


def rates_at_threshold(y_true: ArrayLike, y_score: ArrayLike, threshold: float) -> dict[str, float]:
    """TPR, FPR and alert rate when flagging every score ``>= threshold``."""
    y, s = _as_arrays(y_true, y_score)
    flagged = s >= threshold
    return {
        "tpr": float(flagged[y].mean()) if y.any() else float("nan"),
        "fpr": float(flagged[~y].mean()) if (~y).any() else float("nan"),
        "alert_rate": float(flagged.mean()),
    }


def tpr_at_fpr(y_true: ArrayLike, y_score: ArrayLike, target_fpr: float = 0.05) -> float:
    """Recall at the BAF operating point (primary performance metric)."""
    threshold = threshold_at_fpr(y_true, y_score, target_fpr)
    return rates_at_threshold(y_true, y_score, threshold)["tpr"]


def summary_metrics(
    y_true: ArrayLike, y_score: ArrayLike, target_fpr: float = 0.05
) -> dict[str, float]:
    """All performance metrics reported in the model-comparison tables.

    ``brier`` assumes ``y_score`` is a probability; skip it for uncalibrated scores.
    """
    y, s = _as_arrays(y_true, y_score)
    threshold = threshold_at_fpr(y, s, target_fpr)
    rates = rates_at_threshold(y, s, threshold)
    return {
        "tpr_at_fpr": rates["tpr"],
        "fpr": rates["fpr"],
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "brier": float(brier_score_loss(y, np.clip(s, 0.0, 1.0))),
    }
