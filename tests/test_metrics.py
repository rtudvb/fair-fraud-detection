import numpy as np
import pytest

from frauddetect.evaluation.metrics import (
    rates_at_threshold,
    summary_metrics,
    threshold_at_fpr,
    tpr_at_fpr,
)


def brute_force_best_tpr(y, s, target):
    """Best TPR over every candidate threshold whose FPR is strictly below target."""
    y = y.astype(bool)
    best = 0.0
    for t in np.append(np.unique(s), np.inf):
        flagged = s >= t
        if flagged[~y].mean() < target:
            best = max(best, flagged[y].mean())
    return best


@pytest.mark.parametrize("target", [0.01, 0.05, 0.2])
def test_matches_brute_force(scores, target):
    y, s = scores
    threshold = threshold_at_fpr(y, s, target)
    rates = rates_at_threshold(y, s, threshold)
    assert rates["fpr"] < target
    assert rates["tpr"] == pytest.approx(brute_force_best_tpr(y, s, target))


def test_handles_ties():
    rng = np.random.default_rng(1)
    y = (rng.random(2000) < 0.1).astype(int)
    s = np.round(rng.random(2000) + 0.3 * y, 1)  # heavy ties
    threshold = threshold_at_fpr(y, s, 0.05)
    rates = rates_at_threshold(y, s, threshold)
    assert rates["fpr"] < 0.05
    assert rates["tpr"] == pytest.approx(brute_force_best_tpr(y, s, 0.05))


def test_works_without_positives():
    y = np.zeros(100, dtype=int)
    s = np.linspace(0, 1, 100)
    threshold = threshold_at_fpr(y, s, 0.05)
    assert rates_at_threshold(y, s, threshold)["fpr"] < 0.05


def test_perfect_separation_gives_full_recall():
    y = np.array([0] * 90 + [1] * 10)
    s = np.concatenate([np.linspace(0, 0.5, 90), np.linspace(0.6, 1, 10)])
    assert tpr_at_fpr(y, s, 0.05) == 1.0


def test_summary_metrics_keys_and_ranges(scores):
    y, s = scores
    probs = 1 / (1 + np.exp(-s))
    out = summary_metrics(y, probs)
    assert set(out) == {"tpr_at_fpr", "fpr", "threshold", "roc_auc", "pr_auc", "brier"}
    assert 0.5 < out["roc_auc"] <= 1
    assert out["fpr"] < 0.05


@pytest.mark.parametrize(
    ("y", "s"),
    [([0, 1], [0.1]), ([0, 1], [0.1, np.nan]), ([1, 1], [0.2, 0.3])],
)
def test_invalid_inputs_raise(y, s):
    with pytest.raises(ValueError):
        threshold_at_fpr(np.array(y), np.array(s), 0.05)
