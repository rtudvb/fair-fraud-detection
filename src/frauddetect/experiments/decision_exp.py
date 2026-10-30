"""E7 — three-tier decision model (RQ4, report §4.5; proposal §6.7).

Every policy is chosen on validation month 5 and evaluated on test months 6-7:

=========================  ==========================================================
no_model                   everyone opens normally
all_limited                everyone opens a limited account with enhanced verification
baf_binary_5pct            BAF operating point: top 5% FPR goes to manual review
best_binary                cheapest two-way policy within the capacities
three_tier                 cheapest three-tier policy within the capacities
three_tier_fair            + age-fairness constraint, one threshold pair for everyone
three_tier_fair_groupwise  + age-fairness constraint, one threshold pair per age group
=========================  ==========================================================

The capacities cap manual review (tier 3) and enhanced verification (tier 2). Without the
second cap, cheap verification makes "verify everyone" optimal and the model irrelevant;
``all_limited`` shows that reference point. Comparing the last three policies prices
fairness, and shows whether equal treatment is reachable with a single threshold. All money
figures are scenario assumptions (``configs/costs.yaml``); the sensitivity table changes one
assumption at a time.
"""

from __future__ import annotations

import logging
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from frauddetect.config import load_yaml
from frauddetect.decision.three_tier import (
    TierCosts,
    evaluate_policy,
    group_normalized_scores,
    search_policies,
)
from frauddetect.evaluation.fairness import age_groups
from frauddetect.evaluation.metrics import threshold_at_fpr
from frauddetect.experiments.context import ExperimentContext
from frauddetect.experiments.scoring import get_scores
from frauddetect.viz import TIER_COLORS, TIER_LABELS, use_report_style
from frauddetect.viz.style import AQUA, BLUE, INK, INK_MUTED, ORANGE

log = logging.getLogger(__name__)

POLICY_LABELS = {
    "no_model": "不用模型（全部正常開戶）",
    "all_limited": "不用模型（全部限額＋加強驗證）",
    "baf_binary_5pct": "二元：5% 誤判率門檻轉人工",
    "best_binary": "二元：成本最佳門檻",
    "three_tier": "三級分流",
    "three_tier_fair": "三級分流＋公平限制（單一門檻）",
    "three_tier_fair_groupwise": "三級分流＋公平限制（分組門檻）",
}


def _best(table: pd.DataFrame, *, binary: bool | None = None) -> tuple[float, float] | None:
    candidates = table.loc[table["feasible"]]
    if binary is not None:
        candidates = candidates.loc[candidates["binary"] == binary]
    if candidates.empty:
        return None
    row = candidates.sort_values("total_cost").iloc[0]
    return float(row["t1"]), float(row["t2"])


def choose_and_evaluate(
    valid: dict, test: dict, costs: TierCosts, constraints: dict, rates, target_fpr: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose every policy on ``valid`` and evaluate on ``test``; returns (search, results)."""
    capacity = constraints.get("review_capacity_rate")
    verify_capacity = constraints.get("verify_capacity_rate")
    min_ratio = constraints.get("min_fpr_ratio")
    common = {
        "alert_rates": rates,
        "review_capacity_rate": capacity,
        "verify_capacity_rate": verify_capacity,
    }

    search = search_policies(valid["y"], valid["score"], costs, groups=valid["groups"], **common)
    fair = search_policies(
        valid["y"],
        valid["score"],
        costs,
        groups=valid["groups"],
        min_fpr_ratio=min_ratio,
        **common,
    )
    z_valid = group_normalized_scores(
        valid["score"], valid["groups"], valid["score"], valid["y"], valid["groups"]
    )
    z_test = group_normalized_scores(
        test["score"], test["groups"], valid["score"], valid["y"], valid["groups"]
    )
    fair_groupwise = search_policies(
        valid["y"], z_valid, costs, groups=valid["groups"], min_fpr_ratio=min_ratio, **common
    )

    t_baf = threshold_at_fpr(valid["y"], valid["score"], target_fpr)
    plans = {
        "no_model": ((np.inf, np.inf), test["score"]),
        "all_limited": ((-np.inf, np.inf), test["score"]),
        "baf_binary_5pct": ((t_baf, t_baf), test["score"]),
        "best_binary": (_best(search, binary=True), test["score"]),
        "three_tier": (_best(search), test["score"]),
        "three_tier_fair": (_best(fair), test["score"]),
        "three_tier_fair_groupwise": (_best(fair_groupwise), z_test),
    }
    rows = []
    for name, (thresholds, scores) in plans.items():
        if thresholds is None:
            log.warning("%s: no feasible policy under the constraints", name)
            continue
        row = evaluate_policy(test["y"], scores, *thresholds, costs, groups=test["groups"])
        row["policy"] = name
        row["meets_capacity"] = (capacity is None or row["review_rate"] <= capacity) and (
            verify_capacity is None or row["tier2_rate"] <= verify_capacity
        )
        row["meets_fairness"] = min_ratio is None or row["escalation_fpr_ratio"] >= min_ratio
        rows.append(row)
    results = pd.DataFrame(rows)
    baseline = results.set_index("policy")["total_cost"]
    for ref in ("no_model", "baf_binary_5pct"):
        results[f"saving_vs_{ref}"] = 1 - results["total_cost"] / baseline[ref]
    return search, results


def run(ctx: ExperimentContext) -> dict[str, pd.DataFrame]:
    use_report_style()
    section = ctx.section("decision")
    costs_cfg = load_yaml(ctx.root / "configs" / "costs.yaml")
    costs = TierCosts.from_config(costs_cfg)
    constraints = costs_cfg.get("constraints", {})
    rates = costs_cfg["search"]["alert_rates"]
    cutoff = int(ctx.config["fairness"]["age_cutoff"])

    scored = get_scores(
        ctx, section["model"], protocol="dev", variant=section.get("variant", "base")
    )
    valid, test = (
        {
            "y": part["y"].to_numpy(),
            "score": part["score"].to_numpy(),
            "groups": age_groups(part["age"], cutoff),
        }
        for part in (scored.part("valid"), scored.part("test"))
    )
    search, results = choose_and_evaluate(valid, test, costs, constraints, rates, ctx.target_fpr)
    ctx.save_table(search.head(30), "e7_policy_search_valid")
    ctx.save_table(report_table(results), "e7_policies_test")
    ctx.save_figure(plot_cost_breakdown(results), "e7_cost_breakdown")
    best = results.set_index("policy")
    if "three_tier" in best.index:
        ctx.save_figure(plot_tiers(best.loc["three_tier"]), "e7_tiers")
    ctx.save_figure(plot_cost_surface(search, rates), "e7_cost_surface")

    sensitivity = sensitivity_grid(ctx, costs, costs_cfg, constraints, rates, valid, test)
    ctx.save_table(sensitivity, "e7_sensitivity")
    return {"policies": results, "sensitivity": sensitivity}


def report_table(results: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "policy",
        "tier2_rate",
        "review_rate",
        "fraud_tier1",
        "fraud_tier2",
        "fraud_tier3",
        "cost_per_application",
        "saving_vs_no_model",
        "saving_vs_baf_binary_5pct",
        "escalation_fpr_ratio",
        "meets_capacity",
        "meets_fairness",
    ]
    out = results[cols].copy()
    out["policy"] = out["policy"].map(POLICY_LABELS)
    return out


def sensitivity_grid(ctx, costs, costs_cfg, constraints, rates, valid, test) -> pd.DataFrame:
    """Savings over the BAF binary policy when one assumption changes at a time."""
    grid = costs_cfg.get("sensitivity", {})
    scenarios = [("基準", np.nan, costs, constraints)]
    for m in grid.get("loss_full_multipliers", []):
        scaled = replace(costs, loss_full=costs.loss_full * m, loss_limited=costs.loss_limited * m)
        scenarios.append(("詐騙損失倍數", m, scaled, constraints))
    for v in grid.get("verify_catch", []):
        scenarios.append(("加強驗證攔截率", v, replace(costs, verify_catch=v), constraints))
    for c in grid.get("cost_friction", []):
        scenarios.append(
            ("好客戶被加強驗證的不便成本", c, replace(costs, cost_friction=c), constraints)
        )
    for key, label in (
        ("review_capacity_rate", "人工審核上限"),
        ("verify_capacity_rate", "加強驗證上限"),
    ):
        for k in grid.get(key, []):
            scenarios.append((label, k, costs, {**constraints, key: k}))

    rows = []
    for parameter, value, scenario, limits in scenarios:
        _, results = choose_and_evaluate(valid, test, scenario, limits, rates, ctx.target_fpr)
        by_policy = results.set_index("policy")
        saving = by_policy["saving_vs_baf_binary_5pct"]
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                **{
                    f"saving_{p}": saving.get(p, np.nan)
                    for p in (
                        "all_limited",
                        "best_binary",
                        "three_tier",
                        "three_tier_fair_groupwise",
                    )
                },
                "three_tier_tier2_rate": by_policy["tier2_rate"].get("three_tier", np.nan),
            }
        )
    return pd.DataFrame(rows)


def _per_application(results: pd.DataFrame, column: str) -> pd.Series:
    scale = results["cost_per_application"] / results["total_cost"].replace(0, np.nan)
    return (results[column] * scale).fillna(0)


def plot_cost_breakdown(results: pd.DataFrame):
    data = results.set_index("policy").reindex(
        [p for p in POLICY_LABELS if p in set(results.policy)]
    )
    parts = [
        ("fraud_loss", "詐騙損失", BLUE),
        ("operations", "審核與驗證成本", ORANGE),
        ("customer_cost", "好客戶的不便與流失", AQUA),
    ]
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(data) + 1.4))
    left = np.zeros(len(data))
    for col, label, color in parts:
        values = _per_application(data.reset_index(), col).to_numpy()
        ax.barh(
            range(len(data)),
            values,
            left=left,
            color=color,
            height=0.55,
            label=label,
            edgecolor="white",
            linewidth=1,
        )
        left += values
    for i, total in enumerate(left):
        issues = []
        if not data["meets_capacity"].iloc[i]:
            issues.append("超過人力上限")
        if not data["meets_fairness"].iloc[i]:
            issues.append("未達公平標準")
        note = f"（{'、'.join(issues)}）" if issues else ""
        ax.text(total, i, f"  {total:,.0f}{note}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(data)), [POLICY_LABELS[p] for p in data.index])
    ax.invert_yaxis()
    ax.set(title="各決策方式的期望成本（測試期）", xlabel="每件申請的期望成本（元）")
    ax.set_xlim(0, left.max() * 1.25 if left.max() > 0 else 1)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_tiers(row: pd.Series):
    tier1 = 1 - row["tier2_rate"] - row["review_rate"]
    shares = np.array([tier1, row["tier2_rate"], row["review_rate"]]) * 100
    frauds = np.array([row["fraud_tier1"], row["fraud_tier2"], row["fraud_tier3"]], dtype=float)
    fraud_shares = frauds / max(frauds.sum(), 1) * 100
    labels = [TIER_LABELS[t] for t in (1, 2, 3)]
    colors = [TIER_COLORS[t] for t in (1, 2, 3)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    titles = ("申請件分到各級的比例", "詐騙件分到各級的比例")
    for ax, values, title in zip(axes, (shares, fraud_shares), titles, strict=True):
        ax.bar(range(3), values, color=colors, width=0.6)
        ax.set_xticks(range(3), labels, fontsize=8)
        ax.set(title=title, ylabel="%")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8, color=INK)
    fig.tight_layout()
    return fig


def plot_cost_surface(search: pd.DataFrame, rates):
    """Expected cost per application (validation month) for every rate pair."""
    rates = sorted(set(rates))
    index = {r: i for i, r in enumerate(rates)}
    surface = np.full((len(rates), len(rates)), np.nan)
    escalate = search["tier2_rate"] + search["review_rate"]

    def nearest(value: float) -> int:
        return index[min(rates, key=lambda r: abs(r - value))]

    for esc, rev, cost in zip(
        escalate, search["review_rate"], search["cost_per_application"], strict=True
    ):
        surface[nearest(rev), nearest(esc)] = cost
    cmap = LinearSegmentedColormap.from_list("blue_ramp", ["#cde2fb", "#2a78d6", "#0d366b"])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    image = ax.imshow(surface, origin="lower", cmap=cmap, aspect="auto")
    ticks = [f"{r:.1%}" for r in rates]
    ax.set_xticks(range(len(rates)), ticks, rotation=45, fontsize=7)
    ax.set_yticks(range(len(rates)), ticks, fontsize=7)
    ax.set(
        title="門檻組合的期望成本（驗證期）",
        xlabel="升級比例（二級＋三級）",
        ylabel="轉人工比例（三級）",
    )
    ax.grid(False)
    feasible = search.loc[search["feasible"]]
    if not feasible.empty:
        best = feasible.sort_values("total_cost").iloc[0]
        ax.scatter(
            nearest(best["tier2_rate"] + best["review_rate"]),
            nearest(best["review_rate"]),
            marker="*",
            s=200,
            color="white",
            edgecolor=INK_MUTED,
            zorder=3,
            label="成本最低（符合人力上限）",
        )
        ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(image, ax=ax, label="元／件")
    fig.tight_layout()
    return fig
