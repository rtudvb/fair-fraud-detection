"""Three-tier decision model for account opening (proposal §6.7).

Scores below ``t1`` open normally (tier 1), scores in ``[t1, t2)`` open as a limited
account with enhanced verification (tier 2, mapped to Taiwan's type-3 digital deposit
account), and scores ``>= t2`` go to manual review (tier 3). Setting ``t1 == t2`` removes
tier 2 and gives the usual binary pass / review policy.

The model picks ``(t1, t2)`` on a validation month to minimise expected cost subject to a
manual-review capacity and an age-fairness constraint, then evaluates it on the test months.
Counts are computed from sorted scores, so a full grid search costs ``O(G^2 log n)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class TierCosts:
    """Monetary assumptions of the scenario; see ``configs/costs.yaml``."""

    loss_full: float
    loss_limited: float
    verify_catch: float
    review_catch: float
    cost_verify: float
    cost_review: float
    cost_friction: float
    cost_attrition: float

    def __post_init__(self) -> None:
        for name in ("verify_catch", "review_catch"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a probability.")

    @classmethod
    def from_config(cls, costs_config: Mapping) -> TierCosts:
        return cls(**costs_config["tier_costs"])


def assign_tiers(y_score: ArrayLike, t1: float, t2: float) -> np.ndarray:
    """Tier 1, 2 or 3 for every application."""
    if t1 > t2:
        raise ValueError("t1 must not exceed t2.")
    s = np.asarray(y_score, dtype=float)
    return (1 + (s >= t1) + (s >= t2)).astype(np.int8)


def threshold_for_alert_rate(y_score: ArrayLike, rate: float) -> float:
    """Threshold flagging (about) ``rate`` of all applications (``score >= threshold``)."""
    s = np.sort(np.asarray(y_score, dtype=float))[::-1]
    if not 0 < rate <= 1:
        raise ValueError("rate must lie in (0, 1].")
    k = max(1, int(np.ceil(rate * s.size)))
    return float(s[k - 1])


class _Counter:
    """Counts of scores ``>= t`` per (label, group) from pre-sorted arrays."""

    def __init__(self, y: np.ndarray, s: np.ndarray, g: np.ndarray) -> None:
        self.n = s.size
        self.groups = list(np.unique(g))
        self.fraud = np.sort(s[y])
        self.legit = np.sort(s[~y])
        self.legit_by_group = {name: np.sort(s[(~y) & (g == name)]) for name in self.groups}

    @staticmethod
    def ge(sorted_scores: np.ndarray, t: float) -> int:
        return int(sorted_scores.size - np.searchsorted(sorted_scores, t, side="left"))


def _policy_row(counter: _Counter, t1: float, t2: float, costs: TierCosts) -> dict[str, float]:
    f_ge1, f_ge2 = counter.ge(counter.fraud, t1), counter.ge(counter.fraud, t2)
    l_ge1, l_ge2 = counter.ge(counter.legit, t1), counter.ge(counter.legit, t2)
    fraud = {1: counter.fraud.size - f_ge1, 2: f_ge1 - f_ge2, 3: f_ge2}
    legit = {1: counter.legit.size - l_ge1, 2: l_ge1 - l_ge2, 3: l_ge2}

    fraud_loss = (
        fraud[1] * costs.loss_full
        + fraud[2] * (1 - costs.verify_catch) * costs.loss_limited
        + fraud[3] * (1 - costs.review_catch) * costs.loss_full
    )
    operations = (fraud[2] + legit[2]) * costs.cost_verify + (
        fraud[3] + legit[3]
    ) * costs.cost_review
    customer = legit[2] * costs.cost_friction + legit[3] * costs.cost_attrition

    escalation = {
        name: counter.ge(arr, t1) / arr.size if arr.size else np.nan
        for name, arr in counter.legit_by_group.items()
    }
    values = np.array([v for v in escalation.values() if not np.isnan(v)])
    if values.size == 0:
        fairness = np.nan
    elif values.max() == 0:
        fairness = 1.0
    else:
        fairness = float(values.min() / values.max())

    row = {
        "t1": t1,
        "t2": t2,
        "tier2_rate": (fraud[2] + legit[2]) / counter.n,
        "review_rate": (fraud[3] + legit[3]) / counter.n,
        "fraud_tier1": fraud[1],
        "fraud_tier2": fraud[2],
        "fraud_tier3": fraud[3],
        "legit_tier2": legit[2],
        "legit_tier3": legit[3],
        "fraud_loss": fraud_loss,
        "operations": operations,
        "customer_cost": customer,
        "total_cost": fraud_loss + operations + customer,
        "cost_per_application": (fraud_loss + operations + customer) / counter.n,
        "escalation_fpr_ratio": fairness,
    }
    row.update({f"escalation_rate_{name}": rate for name, rate in escalation.items()})
    return row


def _prepare(y_true: ArrayLike, y_score: ArrayLike, groups: ArrayLike | None) -> _Counter:
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(y_score, dtype=float)
    g = np.full(s.size, "all") if groups is None else np.asarray(groups)
    if not (y.shape == s.shape == g.shape):
        raise ValueError("y_true, y_score and groups must have the same length.")
    return _Counter(y, s, g)


def evaluate_policy(
    y_true: ArrayLike,
    y_score: ArrayLike,
    t1: float,
    t2: float,
    costs: TierCosts,
    groups: ArrayLike | None = None,
) -> dict[str, float]:
    """Cost breakdown, tier rates and escalation fairness of one ``(t1, t2)`` policy."""
    if t1 > t2:
        raise ValueError("t1 must not exceed t2.")
    return _policy_row(_prepare(y_true, y_score, groups), t1, t2, costs)


def search_policies(
    y_true: ArrayLike,
    y_score: ArrayLike,
    costs: TierCosts,
    *,
    groups: ArrayLike | None = None,
    alert_rates: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
    review_capacity_rate: float | None = None,
    verify_capacity_rate: float | None = None,
    min_fpr_ratio: float | None = None,
) -> pd.DataFrame:
    """Evaluate every policy on the alert-rate grid, cheapest feasible policy first.

    ``t1`` is placed at a grid alert rate and ``t2`` at an equal or smaller one, so
    ``t1 <= t2``; equal rates give binary policies. ``feasible`` marks policies that meet
    the review-capacity (tier 3), verification-capacity (tier 2) and fairness constraints
    (a missing constraint always holds).
    """
    counter = _prepare(y_true, y_score, groups)
    s = np.asarray(y_score, dtype=float)
    rates = sorted(set(alert_rates), reverse=True)
    thresholds = {rate: threshold_for_alert_rate(s, rate) for rate in rates}

    rows = []
    for i, escalate_rate in enumerate(rates):
        for review_rate in rates[i:]:
            t1, t2 = thresholds[escalate_rate], thresholds[review_rate]
            if t1 > t2:  # ties in the scores can invert neighbouring quantiles
                continue
            row = _policy_row(counter, t1, t2, costs)
            row["binary"] = escalate_rate == review_rate
            rows.append(row)

    table = pd.DataFrame(rows).drop_duplicates(subset=["t1", "t2"])
    feasible = np.ones(len(table), dtype=bool)
    if review_capacity_rate is not None:
        feasible &= table["review_rate"].to_numpy() <= review_capacity_rate
    if verify_capacity_rate is not None:
        feasible &= table["tier2_rate"].to_numpy() <= verify_capacity_rate
    if min_fpr_ratio is not None:
        feasible &= table["escalation_fpr_ratio"].fillna(0).to_numpy() >= min_fpr_ratio
    table["feasible"] = feasible
    return table.sort_values(["feasible", "total_cost"], ascending=[False, True]).reset_index(
        drop=True
    )


def group_normalized_scores(
    y_score: ArrayLike,
    groups: ArrayLike,
    ref_score: ArrayLike,
    ref_y: ArrayLike,
    ref_groups: ArrayLike,
) -> np.ndarray:
    """Map each score to the share of legitimate *reference* applicants of its group below it.

    Thresholds on these values escalate the same share of legitimate applicants in every
    group, i.e. one threshold pair per group that equalises false positives. The reference
    (validation month) fixes the mapping, so test scores are transformed without peeking.
    Like any per-group threshold, this uses the protected attribute at decision time.
    """
    s = np.asarray(y_score, dtype=float)
    g = np.asarray(groups)
    rs, ry, rg = np.asarray(ref_score, dtype=float), np.asarray(ref_y), np.asarray(ref_groups)
    out = np.empty_like(s)
    for name in np.unique(g):
        reference = np.sort(rs[(rg == name) & (ry == 0)])
        if reference.size == 0:
            raise ValueError(f"no legitimate reference applicants in group {name!r}")
        mask = g == name
        out[mask] = np.searchsorted(reference, s[mask], side="left") / reference.size
    return out


def best_policy(table: pd.DataFrame) -> pd.Series:
    """Cheapest feasible row of :func:`search_policies`."""
    feasible = table.loc[table["feasible"]]
    if feasible.empty:
        raise ValueError("No policy meets the constraints; relax capacity or fairness.")
    return feasible.iloc[0]


__all__ = [
    "TierCosts",
    "assign_tiers",
    "best_policy",
    "evaluate_policy",
    "group_normalized_scores",
    "search_policies",
    "threshold_for_alert_rate",
]
