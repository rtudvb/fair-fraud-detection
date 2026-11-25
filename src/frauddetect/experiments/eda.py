"""E0 — exploratory data analysis of BAF and the Taiwanese fraud context (report §3.1-3.2)."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frauddetect.data.schema import (
    CATEGORICAL_FEATURES,
    LABEL,
    MONTH,
    NEGATIVE_MEANS_MISSING,
    NUMERIC_FEATURES,
    PROTECTED,
)
from frauddetect.experiments.context import ExperimentContext
from frauddetect.viz import GROUP_COLORS, use_report_style
from frauddetect.viz.style import BLUE, INK_SECONDARY, ORANGE

log = logging.getLogger(__name__)


def monthly_overview(df: pd.DataFrame, age_cutoff: int = 50) -> pd.DataFrame:
    older = df[PROTECTED] >= age_cutoff
    return (
        df.assign(older=older)
        .groupby(MONTH)
        .agg(
            applications=(LABEL, "size"),
            frauds=(LABEL, "sum"),
            fraud_rate=(LABEL, "mean"),
            share_older=("older", "mean"),
        )
        .reset_index()
    )


def age_profile(df: pd.DataFrame) -> pd.DataFrame:
    table = df.groupby(PROTECTED).agg(applications=(LABEL, "size"), fraud_rate=(LABEL, "mean"))
    table["share"] = table["applications"] / len(df)
    return table.reset_index()


def missing_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Share of 'missing' sentinels (negative values) and fraud rate with / without them."""
    rows = []
    for col in NEGATIVE_MEANS_MISSING:
        missing = df[col] < 0
        rows.append(
            {
                "feature": col,
                "share_missing": missing.mean(),
                "fraud_rate_missing": df.loc[missing, LABEL].mean() if missing.any() else np.nan,
                "fraud_rate_present": df.loc[~missing, LABEL].mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("share_missing", ascending=False, ignore_index=True)


def categorical_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in CATEGORICAL_FEATURES:
        stats = df.groupby(col, observed=True)[LABEL].agg(["size", "mean"])
        for level, (size, rate) in stats.iterrows():
            rows.append(
                {"feature": col, "level": level, "share": size / len(df), "fraud_rate": rate}
            )
    return pd.DataFrame(rows)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics and Spearman correlation with the label (cf. BAF paper Table 3)."""
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    summary = df[cols].describe().T[["mean", "std", "min", "50%", "max"]]
    summary["spearman_with_fraud"] = [
        df[c].corr(df[LABEL], method="spearman") if (df[c] != df[c].iloc[0]).any() else np.nan
        for c in cols  # constant columns (e.g. device_fraud_count in BAF) have no correlation
    ]
    summary["n_unique"] = [df[c].nunique() for c in cols]
    return (
        summary.reset_index(names="feature")
        .sort_values("spearman_with_fraud", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )


def variant_summary(ctx: ExperimentContext, age_cutoff: int = 50) -> pd.DataFrame:
    """Size and prevalence disparity of every variant (cf. BAF paper Table 1)."""
    rows = []
    for variant in ctx.config["dataset"]["variants"]:
        df = ctx.load(variant, cache=False)
        older = df[PROTECTED] >= age_cutoff
        rows.append(
            {
                "variant": variant,
                "rows": len(df),
                "columns": df.shape[1],
                "fraud_rate": df[LABEL].mean(),
                "share_older": older.mean(),
                "fraud_rate_younger": df.loc[~older, LABEL].mean(),
                "fraud_rate_older": df.loc[older, LABEL].mean(),
            }
        )
    return pd.DataFrame(rows)


def _plot_monthly(table: pd.DataFrame):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    left.bar(table[MONTH], table["applications"] / 1000, color=BLUE, width=0.7)
    left.set(title="每月申請件數", xlabel="月份（0–7）", ylabel="件數（千）")
    right.plot(table[MONTH], table["fraud_rate"] * 100, color=ORANGE, marker="o")
    right.set(title="每月詐騙率", xlabel="月份（0–7）", ylabel="詐騙率（%）")
    right.set_ylim(0, table["fraud_rate"].max() * 100 * 1.25)  # headroom for the label
    for ax in (left, right):
        ax.set_xticks(table[MONTH])
        ax.axvspan(5.5, 7.5, color=INK_SECONDARY, alpha=0.06, lw=0)
        top = ax.get_ylim()[1]
        ax.text(6.5, top * 0.97, "測試期", ha="center", va="top", color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def _plot_age(table: pd.DataFrame, age_cutoff: int = 50):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    colors = [GROUP_COLORS["older" if a >= age_cutoff else "younger"] for a in table[PROTECTED]]
    left.bar(table[PROTECTED], table["share"] * 100, width=7, color=colors)
    left.set(title="申請者年齡分布", xlabel="年齡（每 10 歲一級）", ylabel="占比（%）")
    right.bar(table[PROTECTED], table["fraud_rate"] * 100, width=7, color=colors)
    right.set(title="各年齡層詐騙率", xlabel="年齡（每 10 歲一級）", ylabel="詐騙率（%）")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["younger"]),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["older"]),
    ]
    right.legend(handles, [f"未滿 {age_cutoff} 歲", f"{age_cutoff} 歲以上"], loc="upper left")
    fig.tight_layout()
    return fig


def _plot_missing(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 3))
    ordered = table.sort_values("share_missing")
    ax.barh(ordered["feature"], ordered["share_missing"] * 100, color=BLUE, height=0.6)
    ax.set(title="以負值表示缺值的欄位", xlabel="缺值占比（%）")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def taiwan_context(ctx: ExperimentContext) -> None:
    """Plot hand-compiled Taiwanese fraud statistics if ``data/external`` provides them.

    Expected file: ``data/external/taiwan_fraud_stats.csv`` with columns
    ``period, cases, loss_100m_twd, source`` (template in ``docs/templates/``).
    """
    path = ctx.root / "data" / "external" / "taiwan_fraud_stats.csv"
    if ctx.synthetic or not path.exists():
        log.info("Taiwan statistics not found at %s; skipping (see docs/experiments.md)", path)
        return
    stats = pd.read_csv(path)
    ctx.save_table(stats, "e0_taiwan_fraud_stats", floatfmt=".2f")
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    left.bar(stats["period"].astype(str), stats["cases"] / 10000, color=BLUE, width=0.6)
    left.set(title="詐騙受理件數", ylabel="萬件")
    right.bar(stats["period"].astype(str), stats["loss_100m_twd"], color=ORANGE, width=0.6)
    right.set(title="詐騙財損", ylabel="億元")
    fig.tight_layout()
    ctx.save_figure(fig, "e0_taiwan_fraud_stats")


def run(ctx: ExperimentContext) -> dict[str, pd.DataFrame]:
    use_report_style()
    cutoff = int(ctx.config["fairness"]["age_cutoff"])
    df = ctx.load(ctx.section("eda").get("variant", "base"))
    tables = {
        "e0_monthly": monthly_overview(df, cutoff),
        "e0_age": age_profile(df),
        "e0_missing": missing_profile(df),
        "e0_categorical": categorical_profile(df),
        "e0_numeric": numeric_profile(df),
        "e0_variants": variant_summary(ctx, cutoff),
    }
    for name, table in tables.items():
        ctx.save_table(table, name)
    ctx.save_figure(_plot_monthly(tables["e0_monthly"]), "e0_monthly")
    ctx.save_figure(_plot_age(tables["e0_age"], cutoff), "e0_age")
    ctx.save_figure(_plot_missing(tables["e0_missing"]), "e0_missing")
    taiwan_context(ctx)
    return tables
