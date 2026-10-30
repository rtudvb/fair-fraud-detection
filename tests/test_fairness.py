import numpy as np
import pytest

from frauddetect.evaluation.fairness import (
    OLDER,
    YOUNGER,
    age_groups,
    apply_group_thresholds,
    cutoff_sensitivity,
    fpr_ratio,
    group_rates,
    group_thresholds_at_fpr,
    tpr_ratio,
)


def test_age_groups_inclusive_vs_exclusive():
    age = np.array([40, 50, 60])
    assert list(age_groups(age, 50, inclusive=True)) == [YOUNGER, OLDER, OLDER]
    assert list(age_groups(age, 50, inclusive=False)) == [YOUNGER, YOUNGER, OLDER]


def test_fpr_ratio_known_example():
    # younger: 1 of 10 negatives flagged (FPR 0.1); older: 3 of 10 (FPR 0.3)
    y = np.zeros(20, dtype=int)
    pred = np.array([1] + [0] * 9 + [1, 1, 1] + [0] * 7)
    groups = np.array([YOUNGER] * 10 + [OLDER] * 10)
    rates = group_rates(y, pred, groups)
    assert rates.loc[YOUNGER, "fpr"] == pytest.approx(0.1)
    assert rates.loc[OLDER, "fpr"] == pytest.approx(0.3)
    assert fpr_ratio(y, pred, groups) == pytest.approx(1 / 3)


def test_ratio_is_one_when_no_group_is_flagged():
    y = np.zeros(10, dtype=int)
    groups = np.array([YOUNGER, OLDER] * 5)
    assert fpr_ratio(y, np.zeros(10), groups) == 1.0


def test_tpr_ratio_ignores_groups_without_positives():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 0, 0])
    groups = np.array([YOUNGER, YOUNGER, OLDER, OLDER])
    assert tpr_ratio(y, pred, groups) == 1.0  # only one group has a defined TPR


def test_group_thresholds_meet_target_in_every_group(scores):
    y, s = scores
    groups = np.where(np.arange(len(y)) % 3 == 0, OLDER, YOUNGER)
    s = s + 0.8 * (groups == OLDER)  # older applicants score higher
    thresholds = group_thresholds_at_fpr(y, s, groups, 0.05)
    flags = apply_group_thresholds(s, groups, thresholds)
    rates = group_rates(y, flags, groups)
    assert (rates["fpr"] < 0.05).all()
    assert fpr_ratio(y, flags, groups) > 0.8


def test_apply_group_thresholds_rejects_unknown_group():
    with pytest.raises(KeyError):
        apply_group_thresholds([0.5], ["other"], {YOUNGER: 0.1})


def test_cutoff_sensitivity_has_both_rules_per_cutoff(baf_frame):
    y = baf_frame["fraud_bool"].to_numpy()
    pred = baf_frame["velocity_6h"].to_numpy() > 1.0
    table = cutoff_sensitivity(y, pred, baf_frame["customer_age"], cutoffs=(40, 50, 60))
    assert len(table) == 6
    assert set(table["rule"]) == {">= 40", "> 40", ">= 50", "> 50", ">= 60", "> 60"}
    assert table["fpr_ratio"].between(0, 1).all()
