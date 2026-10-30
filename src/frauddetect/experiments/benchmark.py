"""E2-E4 — three generations of models on BAF Base (RQ2, report §4.2-4.3).

* E2 ``run_default``        library defaults for all ten models, bootstrap CIs, paired tests
* E3 ``run_tuned``          Optuna with the same trial budget for every tunable model
* E4 ``run_learning_curve`` performance and cost as the training set grows (small-bank lens)

A model that fails (missing package, GPU memory, licence) is logged and skipped, so an
overnight run never stops halfway; the failure is listed in the output table.
"""

from __future__ import annotations

import json
import logging
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frauddetect.config import load_yaml
from frauddetect.data.split import split_from_config, stratified_subsample
from frauddetect.evaluation.metrics import tpr_at_fpr
from frauddetect.evaluation.stats import bootstrap_ci, paired_bootstrap_diff
from frauddetect.experiments.context import ExperimentContext
from frauddetect.experiments.scoring import (
    evaluate_scores,
    fit_and_score,
    get_scores,
    runtime_params,
)
from frauddetect.models.registry import MODELS
from frauddetect.models.search import suggest_params
from frauddetect.viz import FAMILY_COLORS, FAMILY_LABELS, MODEL_MARKERS, use_report_style
from frauddetect.viz.style import INK_MUTED, INK_SECONDARY

log = logging.getLogger(__name__)

MODEL_LABELS = {
    "logreg": "邏輯迴歸",
    "random_forest": "隨機森林",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "mlp": "MLP",
    "tabm": "TabM",
    "realmlp": "RealMLP",
    "tabpfn": "TabPFN-3.5",
    "tabicl": "TabICLv2",
}


def _cutoff(ctx: ExperimentContext) -> int:
    return int(ctx.config["fairness"]["age_cutoff"])


def _failure_row(model: str, exc: Exception) -> dict:
    log.error("model %s failed: %s\n%s", model, exc, traceback.format_exc(limit=3))
    return {"model": model, "family": MODELS[model].family, "status": f"failed: {exc}"[:200]}


# ---------------------------------------------------------------------------------- E2
def run_default(ctx: ExperimentContext) -> pd.DataFrame:
    use_report_style()
    section = ctx.section("benchmark")
    variant, n_boot = section.get("variant", "base"), int(section.get("n_boot", 1000))
    rows, test_scores = [], {}
    for model in ctx.models("benchmark"):
        for seed in section.get("seeds", [0]):
            try:
                run = get_scores(ctx, model, protocol="final", variant=variant, seed=seed)
            except Exception as exc:  # keep going with the other models
                rows.append(_failure_row(model, exc) | {"seed": seed})
                continue
            test = run.part("test")
            y, s, age = test["y"].to_numpy(), test["score"].to_numpy(), test["age"].to_numpy()
            metrics = evaluate_scores(y, s, age, target_fpr=ctx.target_fpr, cutoff=_cutoff(ctx))
            if seed == section.get("seeds", [0])[0]:
                test_scores[model] = (y, s)
                ci = bootstrap_ci(
                    lambda yy, ss: tpr_at_fpr(yy, ss, ctx.target_fpr),
                    y,
                    s,
                    n_boot=n_boot,
                    strata=y,
                    seed=ctx.seed,
                )
                metrics |= {"tpr_ci_low": ci.low, "tpr_ci_high": ci.high}
            rows.append(
                {
                    "model": model,
                    "family": MODELS[model].family,
                    "seed": seed,
                    "status": "ok",
                    "n_train": run.n_train,
                    **metrics,
                    "fit_seconds": run.fit_seconds,
                    "predict_seconds": run.predict_seconds,
                }
            )
    results = pd.DataFrame(rows)
    ctx.save_table(results, "e2_default_runs")
    ok = results.loc[results["status"] == "ok"]
    if ok.empty:
        log.error("no model finished; nothing to summarise")
        return results

    summary = _summarise_seeds(ok)
    summary = _add_paired_tests(summary, test_scores, ctx, n_boot)
    ctx.save_table(summary, "e2_default")
    ctx.save_table(
        summary.sort_values("tpr_at_fpr", ascending=False).groupby("family", sort=False).head(1),
        "e2_best_per_family",
    )
    ctx.save_figure(plot_tpr_bars(summary), "e2_tpr")
    ctx.save_figure(plot_cost_vs_performance(summary), "e2_cost_vs_performance")
    ctx.save_figure(plot_fairness_vs_performance(summary), "e2_fairness_vs_performance")
    return summary


def _summarise_seeds(ok: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["tpr_at_fpr", "roc_auc", "pr_auc", "brier", "fpr_ratio", "tpr_ratio"]
    agg = ok.groupby(["model", "family"], sort=False).agg(
        n_seeds=("seed", "size"),
        n_train=("n_train", "first"),
        **{m: (m, "mean") for m in metric_cols},
        tpr_seed_std=("tpr_at_fpr", "std"),
        tpr_ci_low=("tpr_ci_low", "first"),
        tpr_ci_high=("tpr_ci_high", "first"),
        fit_seconds=("fit_seconds", "mean"),
        predict_seconds=("predict_seconds", "mean"),
    )
    return agg.reset_index()


def _add_paired_tests(summary: pd.DataFrame, scores: dict, ctx, n_boot: int) -> pd.DataFrame:
    """TPR difference to the best model with a paired bootstrap p-value."""
    best = summary.loc[summary["tpr_at_fpr"].idxmax(), "model"]
    diffs, pvalues = [], []
    for model in summary["model"]:
        if model == best or model not in scores or best not in scores:
            diffs.append(0.0 if model == best else np.nan)
            pvalues.append(np.nan)
            continue
        y, s_model = scores[model]
        _, s_best = scores[best]
        interval, p = paired_bootstrap_diff(
            lambda yy, ss: tpr_at_fpr(yy, ss, ctx.target_fpr),
            y,
            s_model,
            s_best,
            n_boot=n_boot,
            seed=ctx.seed,
        )
        diffs.append(interval.estimate)
        pvalues.append(p)
    summary = summary.copy()
    summary["best_model"] = best
    summary["tpr_diff_to_best"] = diffs
    summary["p_value_vs_best"] = pvalues
    return summary.sort_values("tpr_at_fpr", ascending=False, ignore_index=True)


def _label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def plot_tpr_bars(summary: pd.DataFrame):
    data = summary.sort_values("tpr_at_fpr")
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(data) + 1.2))
    colors = [FAMILY_COLORS[f] for f in data["family"]]
    ax.barh([_label(m) for m in data["model"]], data["tpr_at_fpr"], color=colors, height=0.6)
    if data["tpr_ci_low"].notna().any():
        ax.errorbar(
            data["tpr_at_fpr"],
            range(len(data)),
            xerr=[
                data["tpr_at_fpr"] - data["tpr_ci_low"],
                data["tpr_ci_high"] - data["tpr_at_fpr"],
            ],
            fmt="none",
            ecolor=INK_SECONDARY,
            elinewidth=1,
            capsize=3,
        )
    for i, value in enumerate(data["tpr_at_fpr"]):
        ax.text(value + 0.005, i + 0.22, f"{value:.3f}", va="center", fontsize=8, color=INK_MUTED)
    ax.set(title="三代模型的偵測率（誤判率 5%，95% 信賴區間）", xlabel="偵測率（TPR@5%FPR）")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _family_legend(  # below the axes, so it never covers a bar label
        ax, data["family"].unique(), loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3
    )
    fig.tight_layout()
    return fig


def plot_cost_vs_performance(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for _, row in summary.iterrows():
        seconds = row["fit_seconds"] + row["predict_seconds"]
        ax.scatter(
            seconds,
            row["tpr_at_fpr"],
            color=FAMILY_COLORS[row["family"]],
            marker=MODEL_MARKERS.get(row["model"], "o"),
            s=60,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            _label(row["model"]),
            (seconds, row["tpr_at_fpr"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xscale("log")
    ax.set(
        title="效能與代價：訓練＋預測時間",
        xlabel="時間（秒，對數尺度）",
        ylabel="偵測率（TPR@5%FPR）",
    )
    _family_legend(ax, summary["family"].unique(), loc="lower right")
    fig.tight_layout()
    return fig


def plot_fairness_vs_performance(summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for _, row in summary.iterrows():
        ax.scatter(
            row["tpr_at_fpr"],
            row["fpr_ratio"],
            color=FAMILY_COLORS[row["family"]],
            marker=MODEL_MARKERS.get(row["model"], "o"),
            s=60,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.annotate(
            _label(row["model"]),
            (row["tpr_at_fpr"], row["fpr_ratio"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color=INK_SECONDARY,
        )
    # Zoom to the data so the labels separate; say where the 0.8 reference sits if it is off-axis.
    low, high = summary["fpr_ratio"].min(), summary["fpr_ratio"].max()
    pad = max(0.03, (high - low) * 0.25)
    ax.set_ylim(max(0, low - pad), min(1.02, high + pad))
    if high + pad >= 0.8:
        ax.axhline(0.8, color=INK_MUTED, lw=0.8, ls="--")
    else:
        ax.text(
            0.99,
            0.97,
            "所有模型皆低於 0.8 的參考標準",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=INK_MUTED,
        )
    ax.set(
        title="效能與年齡公平性（未做公平性處理）",
        xlabel="偵測率（TPR@5%FPR）",
        ylabel="FPR 比（越接近 1 越公平）",
    )
    _family_legend(
        ax, summary["family"].unique(), loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3
    )
    fig.tight_layout()
    return fig


def _family_legend(ax, families, **kwargs) -> None:
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f], label=FAMILY_LABELS[f])
        for f in ("classical", "deep_learning", "foundation")
        if f in set(families)
    ]
    ax.legend(handles=handles, **kwargs)


# ---------------------------------------------------------------------------------- E3
def run_tuned(ctx: ExperimentContext) -> pd.DataFrame:
    import optuna

    use_report_style()
    section = ctx.section("tuned")
    variant = section.get("variant", "base")
    spaces = load_yaml(ctx.root / section["space_file"])
    n_trials = int(section["n_trials"])
    split = split_from_config(ctx.load(variant), ctx.config, "main")
    storage = f"sqlite:///{ctx.out_dir / 'optuna.db'}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rows = []
    for model in ctx.models("tuned"):
        spec = spaces.get(model)
        if spec is None:
            log.warning("no search space for %s; skipped", model)
            continue
        hpo_rows = section.get("hpo_train_rows", {}).get(model) or ctx.max_train_rows(model)
        train = stratified_subsample(split.train, hpo_rows, seed=ctx.seed)

        def objective(trial, model=model, spec=spec, train=train):
            params = {**spec.get("fixed", {}), **suggest_params(trial, spec["space"])}
            run = fit_and_score(
                ctx, model, train, {"valid": split.valid}, params=params, seed=ctx.seed
            )
            valid = run.part("valid")
            trial.set_user_attr("fit_seconds", run.fit_seconds)
            return tpr_at_fpr(valid["y"], valid["score"], ctx.target_fpr)

        study = optuna.create_study(
            study_name=f"{variant}-{model}",
            storage=storage,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=ctx.seed),
            load_if_exists=True,
        )
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        start = time.perf_counter()
        try:
            if completed < n_trials:
                study.optimize(objective, n_trials=n_trials - completed, catch=(Exception,))
            best = {**spec.get("fixed", {}), **_decode_best(study.best_trial, spec["space"])}
            run = get_scores(
                ctx, model, protocol="final", variant=variant, params=best, tag="tuned"
            )
        except Exception as exc:
            rows.append(_failure_row(model, exc))
            continue
        test = run.part("test")
        metrics = evaluate_scores(
            test["y"].to_numpy(),
            test["score"].to_numpy(),
            test["age"].to_numpy(),
            target_fpr=ctx.target_fpr,
            cutoff=_cutoff(ctx),
        )
        rows.append(
            {
                "model": model,
                "family": MODELS[model].family,
                "status": "ok",
                "n_trials": len(study.trials),
                "valid_tpr_best": study.best_value,
                **{f"test_{k}": v for k, v in metrics.items()},
                "hpo_seconds": time.perf_counter() - start,
                "best_params": json.dumps(runtime_params(ctx, model, best), default=str),
            }
        )
    tuned = pd.DataFrame(rows)
    tuned = _attach_default(ctx, tuned)
    ctx.save_table(tuned, "e3_tuned")
    ok = tuned.loc[tuned["status"] == "ok"] if "status" in tuned else tuned
    if not ok.empty and ok["default_tpr_at_fpr"].notna().any():
        ctx.save_figure(plot_default_vs_tuned(ok), "e3_default_vs_tuned")
    return tuned


def _decode_best(trial, space: dict) -> dict:
    """Re-apply list decoding for categorical choices stored as strings."""
    params = dict(trial.params)
    for name, spec in space.items():
        if spec["type"] == "categorical" and any(isinstance(c, list) for c in spec["choices"]):
            params[name] = [int(x) for x in str(params[name]).split("-")]
    return params


def _attach_default(ctx: ExperimentContext, tuned: pd.DataFrame) -> pd.DataFrame:
    """Default-track TPR from E2's table, when E2 has been run."""
    path = ctx.results_root / "benchmark" / "e2_default.csv"
    tuned = tuned.copy()
    tuned["default_tpr_at_fpr"] = np.nan
    if path.exists() and "model" in tuned:
        default = pd.read_csv(path).set_index("model")["tpr_at_fpr"]
        tuned["default_tpr_at_fpr"] = tuned["model"].map(default)
    return tuned


def plot_default_vs_tuned(tuned: pd.DataFrame):
    data = tuned.sort_values("test_tpr_at_fpr")
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(data) + 1.2))
    y = np.arange(len(data))
    for i, (_, row) in enumerate(data.iterrows()):
        color = FAMILY_COLORS[row["family"]]
        ax.plot([row["default_tpr_at_fpr"], row["test_tpr_at_fpr"]], [i, i], color=color, lw=2)
        ax.scatter(row["default_tpr_at_fpr"], i, color="white", edgecolor=color, s=50, zorder=3)
        ax.scatter(row["test_tpr_at_fpr"], i, color=color, s=50, zorder=3)
    ax.set_yticks(y, [_label(m) for m in data["model"]])
    ax.set(title="調參前（空心）與調參後（實心）的偵測率", xlabel="偵測率（TPR@5%FPR）")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    _family_legend(ax, data["family"].unique(), loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------- E4
def run_learning_curve(ctx: ExperimentContext) -> pd.DataFrame:
    use_report_style()
    section = ctx.section("learning_curve")
    curve = ctx.config["learning_curve"]
    split = split_from_config(ctx.load(section.get("variant", "base")), ctx.config, "reproduction")
    test = split.test
    log_path = ctx.out_dir / "curve.csv"
    done = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    finished = {
        (m, int(n), int(r))
        for m, n, r in zip(
            done.get("model", []), done.get("size", []), done.get("repeat", []), strict=False
        )
    }
    full = len(split.train)

    sizes = sorted({full if size is None else min(int(size), full) for size in curve["sizes"]})
    for model in ctx.models("learning_curve"):
        cap = ctx.max_train_rows(model)
        for n in sizes:
            if cap is not None and n > cap:
                log.info("%s: size %d above its cap %d; skipped", model, n, cap)
                continue
            repeats = int(curve.get("repeats_small", 1)) if n <= 10_000 else 1
            for repeat in range(repeats):
                if (model, n, repeat) in finished:
                    continue
                train = stratified_subsample(split.train, n, seed=ctx.seed + repeat)
                try:
                    run = fit_and_score(ctx, model, train, {"test": test}, seed=ctx.seed + repeat)
                except Exception as exc:
                    _failure_row(model, exc)
                    break
                scored = run.part("test")
                metrics = evaluate_scores(
                    scored["y"].to_numpy(),
                    scored["score"].to_numpy(),
                    scored["age"].to_numpy(),
                    target_fpr=ctx.target_fpr,
                    cutoff=_cutoff(ctx),
                )
                row = {
                    "model": model,
                    "family": MODELS[model].family,
                    "size": n,
                    "repeat": repeat,
                    "n_train": run.n_train,
                    **metrics,
                    "fit_seconds": run.fit_seconds,
                }
                pd.DataFrame([row]).to_csv(
                    log_path, mode="a", header=not log_path.exists(), index=False
                )
    if not log_path.exists():
        return pd.DataFrame()
    curve_df = pd.read_csv(log_path)
    curve_df = curve_df.loc[curve_df["model"].isin(ctx.models("learning_curve"))]
    agg = (
        curve_df.groupby(["model", "family", "size"])
        .agg(
            tpr_mean=("tpr_at_fpr", "mean"),
            tpr_min=("tpr_at_fpr", "min"),
            tpr_max=("tpr_at_fpr", "max"),
            fpr_ratio=("fpr_ratio", "mean"),
            fit_seconds=("fit_seconds", "mean"),
            repeats=("repeat", "size"),
        )
        .reset_index()
    )
    ctx.save_table(agg, "e4_learning_curve")
    wide = agg.pivot_table(index="model", columns="size", values="tpr_mean").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    ctx.save_table(wide, "e4_learning_curve_wide")
    ctx.save_figure(plot_learning_curve(agg), "e4_learning_curve")
    return agg


def plot_learning_curve(agg: pd.DataFrame):
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    for model, part in agg.groupby("model"):
        part = part.sort_values("size")
        color = FAMILY_COLORS[part["family"].iloc[0]]
        marker = MODEL_MARKERS.get(model, "o")
        left.plot(part["size"], part["tpr_mean"], color=color, marker=marker, label=_label(model))
        left.fill_between(part["size"], part["tpr_min"], part["tpr_max"], color=color, alpha=0.12)
        right.plot(part["size"], part["fit_seconds"], color=color, marker=marker)
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlabel("訓練資料筆數（對數尺度）")
    right.set_yscale("log")
    left.set(title="資料量與偵測率", ylabel="偵測率（TPR@5%FPR）")
    right.set(title="資料量與訓練時間", ylabel="秒（對數尺度）")
    left.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    return fig
