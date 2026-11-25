"""E1 — reproduce Figure 1 of Jesus et al. (2022) (RQ1, report §4.1).

Random-search LightGBM configurations are trained on months 0-5 and tested on months 6-7 of
each BAF variant. For every model we record recall at 5% FPR and the FPR ratio between age
groups, both with the official code's grouping (``age >= 50``) and the paper text's
(``age > 50``), which is the seed of contribution three (cutoff sensitivity).

The trial log is appended after every model, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from frauddetect.config import load_yaml
from frauddetect.data.schema import PROTECTED
from frauddetect.data.split import split_from_config
from frauddetect.evaluation.fairness import age_groups, fpr_ratio, group_rates
from frauddetect.evaluation.metrics import rates_at_threshold, threshold_at_fpr
from frauddetect.experiments.context import ExperimentContext
from frauddetect.models.registry import build_model, predict_scores
from frauddetect.models.search import sample_params, to_lightgbm_params
from frauddetect.viz import use_report_style
from frauddetect.viz.style import BLUE, GRID, INK_MUTED, ORANGE

log = logging.getLogger(__name__)

# Jesus et al. (2022) report no table, only Figure 1. These ranges were read by eye from the
# opaque points of that figure (the top 5 Base configurations, evaluated on every variant):
# (recall low, recall high, FPR ratio low, FPR ratio high). The text adds that the top Base
# models lie "around 0.3 FPR ratio".
PAPER_FIGURE1 = {
    "base": (0.55, 0.56, 0.32, 0.33),
    "variant_1": (0.52, 0.53, 0.64, 0.71),
    "variant_2": (0.545, 0.56, 0.31, 0.32),
    "variant_3": (0.755, 0.76, 0.35, 0.37),
    "variant_4": (0.425, 0.44, 0.325, 0.33),
    "variant_5": (0.34, 0.375, 0.365, 0.385),
}
VARIANT_NAMES = {
    "base": "Base",
    "variant_1": "Variant I",
    "variant_2": "Variant II",
    "variant_3": "Variant III",
    "variant_4": "Variant IV",
    "variant_5": "Variant V",
}


def trial_configs(space: dict, n: int, seed: int) -> list[tuple[int, dict]]:
    """``n`` (trial_seed, params) pairs; the same list is reused for every variant."""
    rng = np.random.default_rng(seed)
    seeds = rng.choice(1_000_000, size=n, replace=False)
    return [(int(s), sample_params(space, np.random.default_rng(int(s)))) for s in seeds]


def evaluate_trial(
    y: np.ndarray, scores: np.ndarray, age: np.ndarray, target_fpr: float, cutoff: int
) -> dict[str, float]:
    threshold = threshold_at_fpr(y, scores, target_fpr)
    flagged = scores >= threshold
    result = {
        "recall": rates_at_threshold(y, scores, threshold)["tpr"],
        "roc_auc": roc_auc_score(y, scores),
    }
    for rule, inclusive in (("ge", True), ("gt", False)):
        groups = age_groups(age, cutoff, inclusive=inclusive)
        rates = group_rates(y, flagged, groups)
        result[f"fpr_ratio_{rule}{cutoff}"] = fpr_ratio(y, flagged, groups)
        result[f"fpr_older_{rule}{cutoff}"] = rates["fpr"].get("older", np.nan)
        result[f"fpr_younger_{rule}{cutoff}"] = rates["fpr"].get("younger", np.nan)
        result[f"tpr_older_{rule}{cutoff}"] = rates["tpr"].get("older", np.nan)
        result[f"tpr_younger_{rule}{cutoff}"] = rates["tpr"].get("younger", np.nan)
    return result


def run_trials(ctx: ExperimentContext) -> pd.DataFrame:
    section = ctx.section("reproduction")
    space = load_yaml(ctx.root / section["space_file"])["space"]
    n_trials = section["n_trials"]
    configs = trial_configs(space, max(n_trials["base"], n_trials["variants"]), ctx.seed)
    cutoff = int(ctx.config["fairness"]["age_cutoff"])
    n_jobs = ctx.config["experiments"]["runtime"].get("n_jobs", -1)
    cap = section.get("max_estimators")

    log_path = ctx.out_dir / "trials.csv"
    done = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    finished = set(zip(done.get("variant", []), done.get("trial", []), strict=False))

    top_k = int(section.get("top_k", 5))
    for variant in section["variants"]:
        count = n_trials["base"] if variant == "base" else n_trials["variants"]
        wanted = list(range(count))
        if variant != "base" and section.get("variants_include_base_top_k", True):
            # The top-k Base configurations must also run on every variant, otherwise the
            # "top-k" comparison with the paper's Figure 1 rests on whichever of them happen
            # to fall among the first configurations (only one did in the first run).
            log_now = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
            base = log_now.loc[log_now.get("variant", pd.Series(dtype=str)) == "base"]
            if len(base) >= n_trials["base"]:
                top = base.nlargest(top_k, "recall")["trial"].astype(int).tolist()
                wanted += [t for t in top if t not in wanted]
        todo = [t for t in wanted if (variant, t) not in finished]
        if not todo:
            continue
        split = split_from_config(ctx.load(variant, cache=False), ctx.config, "reproduction")
        X_train, y_train = split.xy("train")
        X_test, y_test = split.xy("test")
        age = split.test[PROTECTED].to_numpy()
        for trial in todo:
            trial_seed, params = configs[trial]
            lgbm_params = to_lightgbm_params(params) | {"n_jobs": n_jobs, "verbose": -1}
            if cap:
                lgbm_params["n_estimators"] = min(lgbm_params["n_estimators"], int(cap))
            start = time.perf_counter()
            model = build_model("lightgbm", lgbm_params, random_state=ctx.seed)
            model.fit(X_train, y_train)
            scores = predict_scores(model, X_test)
            row = {
                "variant": variant,
                "trial": trial,
                "trial_seed": trial_seed,
                **evaluate_trial(y_test, scores, age, ctx.target_fpr, cutoff),
                "seconds": round(time.perf_counter() - start, 2),
                **{f"param_{k}": v for k, v in params.items()},
            }
            pd.DataFrame([row]).to_csv(
                log_path, mode="a", header=not log_path.exists(), index=False
            )
            log.info(
                "%s trial %3d: recall=%.3f fpr_ratio=%.3f (%.0fs)",
                variant,
                trial,
                row["recall"],
                row[f"fpr_ratio_ge{cutoff}"],
                row["seconds"],
            )
    trials = pd.read_csv(log_path)
    order = {v: i for i, v in enumerate(section["variants"])}
    return trials.loc[trials["variant"].isin(order)].sort_values(
        ["variant", "trial"], key=lambda s: s.map(order) if s.name == "variant" else s
    )


def summarise(trials: pd.DataFrame, top_k: int, cutoff: int) -> pd.DataFrame:
    """Per variant: recall spread, and recall / FPR ratio of the top-k configurations of Base."""
    base = trials.loc[trials["variant"] == "base"]
    top = base.nlargest(top_k, "recall")["trial"].tolist()
    rows = []
    for variant, part in trials.groupby("variant", sort=False):
        top_part = part.loc[part["trial"].isin(top)]
        rows.append(
            {
                "variant": variant,
                "trials": len(part),
                "recall_median": part["recall"].median(),
                "recall_max": part["recall"].max(),
                "top_k_trials": len(top_part),
                "top_k_recall": top_part["recall"].mean(),
                f"top_k_fpr_ratio_ge{cutoff}": top_part[f"fpr_ratio_ge{cutoff}"].mean(),
                f"top_k_fpr_ratio_gt{cutoff}": top_part[f"fpr_ratio_gt{cutoff}"].mean(),
            }
        )
    return pd.DataFrame(rows)


def comparison_with_paper(summary: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Top-5 recall and FPR ratio next to the ranges read from Figure 1 of the paper.

    The FPR ratio is given for both groupings, so the table shows which one the published
    figure agrees with.
    """

    def span(low: float, high: float) -> str:
        return f"{low:.2f}" if round(low, 2) == round(high, 2) else f"{low:.2f}–{high:.2f}"

    rows = []
    for _, row in summary.iterrows():
        paper = PAPER_FIGURE1.get(row["variant"])
        rows.append(
            {
                "variant": VARIANT_NAMES.get(row["variant"], row["variant"]),
                "paper_recall": span(*paper[:2]) if paper else "—",
                "ours_recall": f"{row['top_k_recall']:.3f}",
                "paper_fpr_ratio": span(*paper[2:]) if paper else "—",
                f"ours_fpr_ratio_ge{cutoff}": f"{row[f'top_k_fpr_ratio_ge{cutoff}']:.3f}",
                f"ours_fpr_ratio_gt{cutoff}": f"{row[f'top_k_fpr_ratio_gt{cutoff}']:.3f}",
            }
        )
    return pd.DataFrame(rows)


def plot_figure1(trials: pd.DataFrame, top_k: int, cutoff: int):
    """Small multiples: one panel per variant, top-k Base configurations highlighted."""
    variants = list(dict.fromkeys(trials["variant"]))
    top = trials.loc[trials["variant"] == "base"].nlargest(top_k, "recall")["trial"]
    ncols = min(3, len(variants))
    nrows = int(np.ceil(len(variants) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.9 * nrows), sharex=True, sharey=True, squeeze=False
    )
    col = f"fpr_ratio_ge{cutoff}"
    for ax, variant in zip(axes.flat, variants, strict=False):
        ax.scatter(trials["recall"], trials[col], s=10, color=GRID, zorder=1)
        part = trials.loc[trials["variant"] == variant]
        highlight = part["trial"].isin(top)
        ax.scatter(part.loc[~highlight, "recall"], part.loc[~highlight, col], s=14, color=BLUE)
        ax.scatter(
            part.loc[highlight, "recall"],
            part.loc[highlight, col],
            s=34,
            color=ORANGE,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.axhline(0.8, color=INK_MUTED, lw=0.8, ls="--")
        ax.set_title(variant.replace("_", " ").title())
        ax.set_ylim(0, 1.02)
    for ax in axes.flat[len(variants) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("偵測率（5% 誤判率下）")
    for ax in axes[:, 0]:
        ax.set_ylabel("FPR 比（年齡公平性）")
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=BLUE, label="該資料集的模型"),
        plt.Line2D([], [], marker="o", ls="", color=ORANGE, label=f"Base 表現前 {top_k} 名的設定"),
        plt.Line2D([], [], marker="o", ls="", color=GRID, label="其他資料集的模型"),
        plt.Line2D([], [], ls="--", color=INK_MUTED, label="0.8 參考線"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def run(ctx: ExperimentContext) -> pd.DataFrame:
    use_report_style()
    section = ctx.section("reproduction")
    cutoff = int(ctx.config["fairness"]["age_cutoff"])
    trials = run_trials(ctx)
    summary = summarise(trials, section["top_k"], cutoff)
    ctx.save_table(summary, "e1_summary")
    if "base" in set(summary["variant"]):
        ctx.save_table(comparison_with_paper(summary, cutoff), "e1_vs_paper")
    ctx.save_figure(plot_figure1(trials, section["top_k"], cutoff), "e1_figure1")
    return summary


__all__ = ["run", "summarise", "trial_configs"]
