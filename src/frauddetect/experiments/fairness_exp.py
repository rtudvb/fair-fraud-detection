"""E5-E6 — age fairness of every model and two mitigations (RQ3, report §4.4).

E5 measures group FPR/TPR at the BAF operating point for all benchmark models and checks
whether the verdict changes with the age cutoff (40/50/60, ``>=`` vs ``>``).

E6 compares, at equal overall alert budgets, (a) one global threshold, (b) one threshold per
age group (post-processing; uses age at decision time) and (c) FairGBM (in-processing; uses
age only during training). Thresholds are chosen on validation month 5 and evaluated on the
test months 6-7, so the reported FPRs are honest out-of-sample numbers.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from frauddetect.data.schema import LABEL, MONTH, PROTECTED
from frauddetect.data.split import split_from_config
from frauddetect.evaluation.fairness import (
    OLDER,
    YOUNGER,
    age_groups,
    apply_group_thresholds,
    cutoff_sensitivity,
    fpr_ratio,
    group_rates,
    group_thresholds_at_fpr,
)
from frauddetect.evaluation.metrics import threshold_at_fpr
from frauddetect.experiments.benchmark import MODEL_LABELS
from frauddetect.experiments.context import ExperimentContext
from frauddetect.experiments.scoring import get_scores
from frauddetect.models.preprocessing import CategoryAligner
from frauddetect.models.registry import MODELS, build_model
from frauddetect.viz import GROUP_COLORS, GROUP_LABELS, use_report_style
from frauddetect.viz.style import AQUA, BLUE, INK, INK_MUTED, ORANGE

log = logging.getLogger(__name__)


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


# ---------------------------------------------------------------------------------- E5
def measure(ctx: ExperimentContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(ctx.config["fairness"]["age_cutoff"])
    cutoffs = tuple(ctx.config["fairness"]["cutoff_sensitivity"])
    variant = ctx.section("fairness").get("variant", "base")
    rate_rows, sensitivity = [], []
    for model in ctx.models("benchmark"):
        try:
            test = get_scores(ctx, model, protocol="final", variant=variant).part("test")
        except Exception as exc:
            log.error("no scores for %s (%s); skipped", model, exc)
            continue
        y, s, age = test["y"].to_numpy(), test["score"].to_numpy(), test["age"].to_numpy()
        flagged = s >= threshold_at_fpr(y, s, ctx.target_fpr)
        groups = age_groups(age, cutoff)
        rates = group_rates(y, flagged, groups)
        rate_rows.append(
            {
                "model": model,
                "family": MODELS[model].family,
                "fpr_younger": rates.loc[YOUNGER, "fpr"],
                "fpr_older": rates.loc[OLDER, "fpr"],
                "tpr_younger": rates.loc[YOUNGER, "tpr"],
                "tpr_older": rates.loc[OLDER, "tpr"],
                "fpr_ratio": fpr_ratio(y, flagged, groups),
                "older_flag_multiple": rates.loc[OLDER, "fpr"] / rates.loc[YOUNGER, "fpr"],
            }
        )
        table = cutoff_sensitivity(y, flagged, age, cutoffs)
        table.insert(0, "model", model)
        sensitivity.append(table)
    rates_df = pd.DataFrame(rate_rows).sort_values("fpr_ratio", ignore_index=True)
    sens_df = pd.concat(sensitivity, ignore_index=True) if sensitivity else pd.DataFrame()
    return rates_df, sens_df


def plot_group_fpr(rates: pd.DataFrame):
    data = rates.sort_values("fpr_ratio", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(data) + 1.3))
    y = np.arange(len(data))
    height = 0.36
    ax.barh(
        y - height / 2,
        data["fpr_younger"] * 100,
        height,
        color=GROUP_COLORS[YOUNGER],
        label=GROUP_LABELS[YOUNGER],
    )
    ax.barh(
        y + height / 2,
        data["fpr_older"] * 100,
        height,
        color=GROUP_COLORS[OLDER],
        label=GROUP_LABELS[OLDER],
    )
    for i, ratio in enumerate(data["fpr_ratio"]):
        right = max(data["fpr_younger"].iloc[i], data["fpr_older"].iloc[i]) * 100
        ax.text(right + 0.2, i, f"FPR 比 {ratio:.2f}", va="center", fontsize=8, color=INK_MUTED)
    ax.set_yticks(y, [_label(m) for m in data["model"]])
    ax.invert_yaxis()
    ax.set(title="各年齡組的誤判率（整體誤判率 5%）", xlabel="誤判率（%）")
    ax.set_xlim(0, data[["fpr_younger", "fpr_older"]].to_numpy().max() * 100 * 1.22)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    return fig


def plot_cutoff_sensitivity(sens: pd.DataFrame, rates: pd.DataFrame):
    """FPR ratio of every model under every distinct age grouping, as an annotated heatmap.

    BAF ages are whole decades, so "> 40" and ">= 50" select the same people; groupings are
    de-duplicated by the share of applicants they call older.
    """
    order = rates.sort_values("fpr_ratio", ascending=False)["model"].tolist()
    columns = (
        sens.assign(inclusive=sens["rule"].str.startswith(">="))
        .sort_values(["share_older", "inclusive"], ascending=[False, False])
        .drop_duplicates("share_older")
    )
    labels = [
        f"{int(c)} 歲以上" if inc else f"大於 {int(c)} 歲"
        for c, inc in zip(columns["cutoff"], columns["inclusive"], strict=True)
    ]
    labels = [
        f"{label}\n（占 {share:.0%}）"
        for label, share in zip(labels, columns["share_older"], strict=True)
    ]
    grid = np.array(
        [
            [
                sens.loc[
                    (sens["model"] == m) & np.isclose(sens["share_older"], share), "fpr_ratio"
                ].iloc[0]
                for share in columns["share_older"]
            ]
            for m in order
        ]
    )
    cmap = LinearSegmentedColormap.from_list("blue_ramp", ["#0d366b", "#2a78d6", "#cde2fb"])
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(order) + 1.6))
    image = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(
                j,
                i,
                f"{grid[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if grid[i, j] < 0.5 else INK,
            )
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_yticks(range(len(order)), [_label(m) for m in order], fontsize=8)
    ax.set(title="年齡分組方式與 FPR 比（年長組定義）")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="FPR 比（越接近 1 越公平）")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------- E6
def postprocessing(ctx: ExperimentContext, model: str, variant: str) -> pd.DataFrame:
    """Global vs per-group thresholds, chosen on month 5, at several overall FPR targets."""
    run = get_scores(ctx, model, protocol="dev", variant=variant)
    valid, test = run.part("valid"), run.part("test")
    cutoff = int(ctx.config["fairness"]["age_cutoff"])
    g_valid, g_test = age_groups(valid["age"], cutoff), age_groups(test["age"], cutoff)
    rows = []
    for target in ctx.section("fairness").get("operating_fprs", [ctx.target_fpr]):
        t_global = threshold_at_fpr(valid["y"], valid["score"], target)
        t_groups = group_thresholds_at_fpr(valid["y"], valid["score"], g_valid, target)
        flags = {
            "global_threshold": test["score"].to_numpy() >= t_global,
            "group_thresholds": apply_group_thresholds(test["score"], g_test, t_groups),
        }
        for method, flagged in flags.items():
            rows.append(_rates_row(model, method, target, test["y"].to_numpy(), flagged, g_test))
    return pd.DataFrame(rows)


def _rates_row(model, method, target, y, flagged, groups) -> dict:
    rates = group_rates(y, flagged, groups)
    y = np.asarray(y).astype(bool)
    return {
        "model": model,
        "method": method,
        "target_fpr": target,
        "tpr": flagged[y].mean(),
        "fpr": flagged[~y].mean(),
        "fpr_younger": rates["fpr"].get(YOUNGER, np.nan),
        "fpr_older": rates["fpr"].get(OLDER, np.nan),
        "fpr_ratio": fpr_ratio(y, flagged, groups),
        "tpr_younger": rates["tpr"].get(YOUNGER, np.nan),
        "tpr_older": rates["tpr"].get(OLDER, np.nan),
    }


def fairgbm_runs(ctx: ExperimentContext, variant: str) -> pd.DataFrame:
    """FairGBM grid plus an unconstrained LightGBM with the same base parameters."""
    try:
        from fairgbm import FairGBMClassifier
    except ImportError:
        log.warning("fairgbm not installed (pip install -e '.[fair]'); in-processing skipped")
        return pd.DataFrame()
    section = ctx.section("fairness")["fairgbm"]
    cutoff = int(ctx.config["fairness"]["age_cutoff"])
    split = split_from_config(ctx.load(variant), ctx.config, "main")
    X_train, y_train = split.xy("train")
    aligner = CategoryAligner().fit(X_train)
    parts = {"valid": split.valid, "test": split.test}

    def frame(part: pd.DataFrame) -> pd.DataFrame:
        return aligner.transform(part.drop(columns=[LABEL, MONTH]))

    base = dict(section["params"])
    candidates = [("lightgbm_same_params", None)] + [
        (f"fairgbm_tol{c['constraint_fpr_tolerance']}_mlr{c['multiplier_learning_rate']}", c)
        for c in section.get("grid", [])
    ]
    constraint = dict(section.get("constraint", {}))
    rows = []
    for name, extra in candidates:
        try:
            if extra is None:
                model = build_model("lightgbm", base | {"verbose": -1}, random_state=ctx.seed).fit(
                    X_train, y_train
                )
                scores = {
                    k: model.predict_proba(v.drop(columns=[LABEL, MONTH]))[:, 1]
                    for k, v in parts.items()
                }
                if constraint.get("score_threshold") == "auto":
                    # FairGBM enforces its constraints at `score_threshold` (default 0.5). With
                    # ~1% fraud almost nobody is flagged at 0.5, so equal group FPRs hold
                    # trivially and the constraint does nothing. Enforce it at the operating
                    # point instead: the unconstrained model's training threshold at the target FPR.
                    constraint["score_threshold"] = threshold_at_fpr(
                        y_train, model.predict_proba(X_train)[:, 1], ctx.target_fpr
                    )
                    log.info("FairGBM score_threshold = %.4f", constraint["score_threshold"])
            else:
                model = FairGBMClassifier(**base, **constraint, **extra, random_state=ctx.seed)
                group = (split.train[PROTECTED] >= cutoff).astype(int).to_numpy()
                model.fit(
                    aligner.transform(X_train), y_train, constraint_group=group, verbose=False
                )
                scores = {k: model.predict_proba(frame(v))[:, 1] for k, v in parts.items()}
        except Exception as exc:
            log.error("%s failed: %s", name, exc)
            continue
        y_valid = split.valid[LABEL].to_numpy()
        y_test = split.test[LABEL].to_numpy()
        g_test = age_groups(split.test[PROTECTED], cutoff)
        for target in ctx.section("fairness").get("operating_fprs", [ctx.target_fpr]):
            threshold = threshold_at_fpr(y_valid, scores["valid"], target)
            flagged = scores["test"] >= threshold
            rows.append(_rates_row(name, "in_processing", target, y_test, flagged, g_test))
    return pd.DataFrame(rows)


FAIRGBM_STYLES = {  # tolerance -> (colour, marker, line); darker means a looser constraint
    0.02: ("#0b4f3a", "^", (0, (1, 1.5))),
    0.01: ("#1d8f66", "v", (0, (4, 1.5))),
    0.005: ("#6cc4a0", "s", (0, (6, 2, 1, 2))),
    0.0: ("#a9a9a9", "D", (0, (2, 3))),
}


def _tradeoff_style(model: str, method: str) -> tuple[str, str, str]:
    """(color, marker, label): color encodes the method, marker the model."""
    if method == "global_threshold":
        return BLUE, "o" if model != "tabpfn" else "s", f"{_label(model)}：單一門檻"
    if method == "group_thresholds":
        return ORANGE, "o" if model != "tabpfn" else "s", f"{_label(model)}：分組門檻"
    if model == "lightgbm_same_params":
        return BLUE, "D", "LightGBM（與 FairGBM 同參數，未限制）"
    tolerance = model.split("_tol")[-1].split("_mlr")[0]
    # one marker and shade per tolerance, so the four FairGBM lines can be told apart
    shade, marker, _ = FAIRGBM_STYLES.get(float(tolerance), (AQUA, "^", ":"))
    return shade, marker, f"FairGBM（容許差距 {tolerance}）"


def _tradeoff_line(model: str, method: str, default: str):
    """Line style: per tolerance for FairGBM, so its four lines differ in more than shade."""
    if method == "in_processing" and "_tol" in model:
        tolerance = float(model.split("_tol")[-1].split("_mlr")[0])
        return FAIRGBM_STYLES.get(tolerance, (None, None, default))[2]
    return default


def plot_tradeoff(table: pd.DataFrame, target: float):
    """TPR vs FPR ratio across operating points; the target point of each line is ringed."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    styles = {"global_threshold": "-", "group_thresholds": "--", "in_processing": ":"}
    for (model, method), part in table.groupby(["model", "method"], sort=False):
        color, marker, label = _tradeoff_style(model, method)
        part = part.sort_values("target_fpr")
        ax.plot(
            part["tpr"],
            part["fpr_ratio"],
            ls=_tradeoff_line(model, method, styles[method]),
            color=color,
            marker=marker,
            markersize=5,
            label=label,
        )
        at_target = part.loc[np.isclose(part["target_fpr"], target)]
        ax.scatter(
            at_target["tpr"],
            at_target["fpr_ratio"],
            s=110,
            facecolor="none",
            edgecolor=color,
            linewidth=1.5,
            zorder=3,
        )
    ax.axhline(0.8, color=INK_MUTED, lw=0.8, ls=":")
    ax.set_ylim(0, 1.05)
    ax.set(
        title=f"公平性與偵測率的取捨（圈起處為整體誤判率 {target:.0%}）",
        xlabel="偵測率（測試期）",
        ylabel="FPR 比（越接近 1 越公平）",
    )
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))  # keep data visible
    fig.tight_layout()
    return fig


def run(ctx: ExperimentContext) -> dict[str, pd.DataFrame]:
    use_report_style()
    section = ctx.section("fairness")
    variant = section.get("variant", "base")
    rates, sens = measure(ctx)
    outputs = {"e5_group_rates": rates, "e5_cutoff_sensitivity": sens}
    if not rates.empty:
        ctx.save_table(rates, "e5_group_rates")
        ctx.save_table(sens, "e5_cutoff_sensitivity")
        ctx.save_figure(plot_group_fpr(rates), "e5_group_fpr")
        ctx.save_figure(plot_cutoff_sensitivity(sens, rates), "e5_cutoff_sensitivity")

    mitigation = []
    for model in section.get("mitigation_models", []):
        try:
            mitigation.append(postprocessing(ctx, model, variant))
        except Exception as exc:
            log.error("post-processing for %s failed: %s", model, exc)
    mitigation.append(fairgbm_runs(ctx, variant))
    table = pd.concat([m for m in mitigation if not m.empty], ignore_index=True)
    if not table.empty:
        ctx.save_table(table, "e6_mitigation")
        at_target = table.loc[np.isclose(table["target_fpr"], ctx.target_fpr)]
        ctx.save_table(at_target, "e6_mitigation_at_5pct")
        ctx.save_figure(plot_tradeoff(table, ctx.target_fpr), "e6_tradeoff")
    outputs["e6_mitigation"] = table
    return outputs
