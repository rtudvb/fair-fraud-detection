import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from frauddetect.evaluation.metrics import tpr_at_fpr
from frauddetect.evaluation.stats import bootstrap_ci, paired_bootstrap_diff


def test_bootstrap_interval_brackets_estimate(scores):
    y, s = scores
    ci = bootstrap_ci(roc_auc_score, y, s, n_boot=200, strata=y, seed=1)
    assert ci.low <= ci.estimate <= ci.high
    assert ci.high - ci.low < 0.2


def test_bootstrap_is_reproducible(scores):
    y, s = scores
    a = bootstrap_ci(tpr_at_fpr, y, s, n_boot=50, strata=y, seed=3)
    b = bootstrap_ci(tpr_at_fpr, y, s, n_boot=50, strata=y, seed=3)
    assert a == b


def test_stratified_bootstrap_keeps_prevalence(scores):
    y, _ = scores
    ci = bootstrap_ci(np.mean, y, n_boot=50, strata=y)
    assert ci.low == pytest.approx(y.mean())
    assert ci.high == pytest.approx(y.mean())


def test_paired_difference_of_identical_models_is_zero(scores):
    y, s = scores
    ci, p_value = paired_bootstrap_diff(roc_auc_score, y, s, s, n_boot=50)
    assert ci.estimate == ci.low == ci.high == 0
    assert p_value == 1.0


def test_paired_difference_detects_a_better_model(scores):
    y, s = scores
    noisy = s + np.random.default_rng(5).normal(scale=2.0, size=len(s))
    ci, p_value = paired_bootstrap_diff(roc_auc_score, y, s, noisy, n_boot=200)
    assert ci.low > 0
    assert p_value < 0.05


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(roc_auc_score, [0, 1], [0.1, 0.2, 0.3])
