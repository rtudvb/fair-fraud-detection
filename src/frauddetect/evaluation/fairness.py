"""Group fairness for fraud flags (predictive equality across age groups).

A false positive denies a legitimate applicant access to a bank account, so the primary
fairness metric is the ratio of group FPRs (predictive equality), as in Jesus et al. (2022):
``min(FPR_g) / max(FPR_g)``; 1.0 is perfect parity, values below 0.8 are flagged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from frauddetect.evaluation.metrics import threshold_at_fpr

OLDER = "older"
YOUNGER = "younger"


def age_groups(age: ArrayLike, cutoff: int = 50, *, inclusive: bool = True) -> np.ndarray:
    """Label each applicant ``"older"`` or ``"younger"``.

    ``inclusive=True`` reproduces the official BAF code (``age >= cutoff``); the paper text
    describes ``age > cutoff``. BAF ages are rounded to decades, so the two choices move the
    whole 50-year-old bin between groups.
    """
    age = np.asarray(age)
    older = age >= cutoff if inclusive else age > cutoff
    return np.where(older, OLDER, YOUNGER)


def group_rates(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> pd.DataFrame:
    """Per-group size, prevalence, FPR, TPR and flag rate for binary predictions."""
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true).astype(bool),
            "pred": np.asarray(y_pred).astype(bool),
            "group": np.asarray(groups),
        }
    )
    rows = []
    for name, part in frame.groupby("group", sort=True):
        negatives, positives = part.loc[~part["y"], "pred"], part.loc[part["y"], "pred"]
        rows.append(
            {
                "group": name,
                "n": len(part),
                "prevalence": part["y"].mean(),
                "fpr": negatives.mean() if len(negatives) else np.nan,
                "tpr": positives.mean() if len(positives) else np.nan,
                "flag_rate": part["pred"].mean(),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def _ratio(values: pd.Series) -> float:
    values = values.dropna()
    if values.empty:
        return float("nan")
    high, low = values.max(), values.min()
    if high == 0:
        return 1.0  # all groups at zero: parity
    return float(low / high)


def fpr_ratio(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Predictive equality: lowest group FPR divided by highest group FPR."""
    return _ratio(group_rates(y_true, y_pred, groups)["fpr"])


def tpr_ratio(y_true: ArrayLike, y_pred: ArrayLike, groups: ArrayLike) -> float:
    """Equal opportunity: lowest group TPR divided by highest group TPR."""
    return _ratio(group_rates(y_true, y_pred, groups)["tpr"])


def group_thresholds_at_fpr(
    y_true: ArrayLike, y_score: ArrayLike, groups: ArrayLike, target_fpr: float = 0.05
) -> dict[str, float]:
    """Post-processing mitigation: one threshold per group, each meeting ``target_fpr``.

    This equalises group FPRs by construction but uses the protected attribute at decision
    time, which the report discusses as a possible form of disparate treatment.
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=float)
    g = np.asarray(groups)
    return {name: threshold_at_fpr(y[g == name], s[g == name], target_fpr) for name in np.unique(g)}


def apply_group_thresholds(
    y_score: ArrayLike, groups: ArrayLike, thresholds: dict[str, float]
) -> np.ndarray:
    """Binary flags from per-group thresholds (``score >= threshold`` of the row's group)."""
    s = np.asarray(y_score, dtype=float)
    g = np.asarray(groups)
    unknown = set(np.unique(g)) - set(thresholds)
    if unknown:
        raise KeyError(f"No threshold for groups {sorted(unknown)}")
    cut = np.vectorize(thresholds.__getitem__, otypes=[float])(g)
    return s >= cut


def cutoff_sensitivity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    age: ArrayLike,
    cutoffs: tuple[int, ...] = (40, 50, 60),
) -> pd.DataFrame:
    """FPR ratio and group FPRs for several age cutoffs, inclusive and exclusive (H3)."""
    rows = []
    for cutoff in cutoffs:
        for inclusive in (True, False):
            groups = age_groups(age, cutoff, inclusive=inclusive)
            rates = group_rates(y_true, y_pred, groups)
            rows.append(
                {
                    "cutoff": cutoff,
                    "rule": f">= {cutoff}" if inclusive else f"> {cutoff}",
                    "share_older": float(np.mean(groups == OLDER)),
                    "fpr_older": rates["fpr"].get(OLDER, np.nan),
                    "fpr_younger": rates["fpr"].get(YOUNGER, np.nan),
                    "fpr_ratio": _ratio(rates["fpr"]),
                }
            )
    return pd.DataFrame(rows)
