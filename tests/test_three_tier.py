import numpy as np
import pytest

from frauddetect.config import load_yaml, project_root
from frauddetect.decision.three_tier import (
    TierCosts,
    assign_tiers,
    best_policy,
    evaluate_policy,
    group_normalized_scores,
    search_policies,
    threshold_for_alert_rate,
)

COSTS = TierCosts(
    loss_full=200_000,
    loss_limited=50_000,
    verify_catch=0.5,
    review_catch=0.9,
    cost_verify=50,
    cost_review=300,
    cost_friction=100,
    cost_attrition=2000,
)


def naive_cost(y, s, t1, t2, c):
    tiers = assign_tiers(s, t1, t2)
    fraud, legit = y == 1, y == 0
    n = {
        (lab, t): int(np.sum(mask & (tiers == t)))
        for lab, mask in (("f", fraud), ("l", legit))
        for t in (1, 2, 3)
    }
    return (
        n["f", 1] * c.loss_full
        + n["f", 2] * (1 - c.verify_catch) * c.loss_limited
        + n["f", 3] * (1 - c.review_catch) * c.loss_full
        + (n["f", 2] + n["l", 2]) * c.cost_verify
        + (n["f", 3] + n["l", 3]) * c.cost_review
        + n["l", 2] * c.cost_friction
        + n["l", 3] * c.cost_attrition
    )


def test_assign_tiers():
    assert list(assign_tiers([0.1, 0.5, 0.9], 0.4, 0.8)) == [1, 2, 3]
    with pytest.raises(ValueError):
        assign_tiers([0.1], 0.9, 0.4)


def test_evaluate_policy_matches_naive_count(scores):
    y, s = scores
    t1, t2 = np.quantile(s, [0.9, 0.98])
    row = evaluate_policy(y, s, t1, t2, COSTS)
    assert row["total_cost"] == pytest.approx(naive_cost(y, s, t1, t2, COSTS))
    tiers = assign_tiers(s, t1, t2)
    assert row["fraud_tier3"] == np.sum((y == 1) & (tiers == 3))
    assert row["review_rate"] == pytest.approx(np.mean(tiers == 3))


def test_binary_policy_has_no_tier_two(scores):
    y, s = scores
    t = np.quantile(s, 0.95)
    row = evaluate_policy(y, s, t, t, COSTS)
    assert row["tier2_rate"] == 0
    assert row["fraud_tier2"] == row["legit_tier2"] == 0


def test_alert_rate_threshold():
    s = np.arange(1000, dtype=float)
    assert np.mean(s >= threshold_for_alert_rate(s, 0.05)) == pytest.approx(0.05)


def test_search_respects_constraints(scores):
    y, s = scores
    groups = np.where(np.arange(len(y)) % 4 == 0, "older", "younger")
    table = search_policies(
        y,
        s,
        COSTS,
        groups=groups,
        alert_rates=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
        review_capacity_rate=0.02,
        min_fpr_ratio=0.8,
    )
    best = best_policy(table)
    assert best["review_rate"] <= 0.02
    assert best["escalation_fpr_ratio"] >= 0.8
    assert (table.loc[table["feasible"], "total_cost"] >= best["total_cost"]).all()
    assert table["binary"].any() and (~table["binary"]).any()


def test_three_tiers_beat_binary_when_limits_are_cheap(scores):
    y, s = scores
    table = search_policies(y, s, COSTS, alert_rates=(0.01, 0.02, 0.05, 0.1, 0.2))
    best_three = table.loc[~table["binary"], "total_cost"].min()
    best_binary = table.loc[table["binary"], "total_cost"].min()
    assert best_three <= best_binary


def test_infeasible_constraints_raise(scores):
    y, s = scores
    table = search_policies(y, s, COSTS, alert_rates=(0.1, 0.2), review_capacity_rate=0.01)
    with pytest.raises(ValueError):
        best_policy(table)


def test_costs_config_loads():
    costs = TierCosts.from_config(load_yaml(project_root() / "configs" / "costs.yaml"))
    assert costs.loss_limited < costs.loss_full


def test_probabilities_are_validated():
    with pytest.raises(ValueError):
        TierCosts(1, 1, 1.5, 0.9, 1, 1, 1, 1)


def test_group_normalized_scores_equalise_legit_escalation():
    rng = np.random.default_rng(3)
    n = 80000
    groups = np.where(rng.random(n) < 0.25, "older", "younger")
    y = (rng.random(n) < 0.02).astype(int)
    s = rng.normal(size=n) + 0.8 * (groups == "older") + 2 * y  # older score higher
    half = n // 2
    z = group_normalized_scores(s[half:], groups[half:], s[:half], y[:half], groups[:half])
    legit = y[half:] == 0
    rates = {g: np.mean(z[legit & (groups[half:] == g)] >= 0.95) for g in ("older", "younger")}
    assert rates["older"] == pytest.approx(rates["younger"], abs=0.01)
    assert rates["older"] == pytest.approx(0.05, abs=0.01)
    raw = {
        g: np.mean(s[half:][legit & (groups[half:] == g)] >= np.quantile(s[:half], 0.95))
        for g in ("older", "younger")
    }
    assert raw["older"] > 2 * raw["younger"]  # without normalisation the gap is large


def test_group_normalized_scores_need_reference_group():
    with pytest.raises(ValueError):
        group_normalized_scores([0.1], ["a"], [0.2], [0], ["b"])


def test_search_respects_verification_capacity(scores):
    y, s = scores
    rates = (0.01, 0.02, 0.05, 0.1, 0.2)
    table = search_policies(y, s, COSTS, alert_rates=rates, verify_capacity_rate=0.05)
    feasible = table.loc[table["feasible"]]
    assert (feasible["tier2_rate"] <= 0.05).all()
    assert (~table["feasible"]).any()  # policies with more verification are ruled out
    assert best_policy(table)["tier2_rate"] <= 0.05
