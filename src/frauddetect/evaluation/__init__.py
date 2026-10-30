"""Performance, fairness and statistical evaluation."""

from frauddetect.evaluation.fairness import (
    age_groups,
    apply_group_thresholds,
    cutoff_sensitivity,
    fpr_ratio,
    group_rates,
    group_thresholds_at_fpr,
    tpr_ratio,
)
from frauddetect.evaluation.metrics import (
    rates_at_threshold,
    summary_metrics,
    threshold_at_fpr,
    tpr_at_fpr,
)
from frauddetect.evaluation.stats import Interval, bootstrap_ci, paired_bootstrap_diff

__all__ = [
    "Interval",
    "age_groups",
    "apply_group_thresholds",
    "bootstrap_ci",
    "cutoff_sensitivity",
    "fpr_ratio",
    "group_rates",
    "group_thresholds_at_fpr",
    "paired_bootstrap_diff",
    "rates_at_threshold",
    "summary_metrics",
    "threshold_at_fpr",
    "tpr_at_fpr",
    "tpr_ratio",
]
