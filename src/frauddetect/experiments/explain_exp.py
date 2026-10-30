"""E8 — what drives the fraud score, as SHAP values and readable rules (RQ5, report §4.6).

The explained model is refitted on months 0-5 with its default parameters (identical to the
E2 model), TreeSHAP is computed on a sample of the test months, and surrogate trees of a few
depths translate the model's flags at the BAF operating point into rules.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from frauddetect.data.split import split_from_config
from frauddetect.evaluation.metrics import threshold_at_fpr
from frauddetect.experiments.context import ExperimentContext
from frauddetect.experiments.scoring import default_params, runtime_params
from frauddetect.explain.rules import fit_surrogate, leaf_rules
from frauddetect.models.registry import build_model, predict_scores
from frauddetect.viz import use_report_style
from frauddetect.viz.style import BLUE, INK_MUTED

log = logging.getLogger(__name__)

TREE_MODELS = {"lightgbm", "xgboost", "catboost", "random_forest"}


def shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """TreeSHAP values (log-odds contributions) of the positive class for a fitted pipeline."""
    import shap

    aligned = model.named_steps["align"].transform(X)
    estimator = model.named_steps["clf"]
    if hasattr(estimator, "model_"):  # CatBoost wrapper
        estimator = estimator.model_
        aligned = model.named_steps["clf"]._to_catboost(aligned)
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(aligned)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 else values


def importance_table(values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for j, col in enumerate(X.columns):
        feature = X[col]
        if isinstance(feature.dtype, pd.CategoricalDtype):
            direction = ""
        else:
            numeric = feature.to_numpy(dtype=float)
            corr = np.corrcoef(numeric, values[:, j])[0, 1] if numeric.std() > 0 else np.nan
            direction = (
                "數值越高越可疑" if corr > 0.1 else ("數值越低越可疑" if corr < -0.1 else "非單調")
            )
        rows.append(
            {"feature": col, "mean_abs_shap": np.abs(values[:, j]).mean(), "direction": direction}
        )
    table = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def plot_importance(table: pd.DataFrame, top: int = 15):
    data = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(data) + 1.2))
    ax.barh(data["feature"], data["mean_abs_shap"], color=BLUE, height=0.6)
    ax.set(
        title=f"影響詐騙分數最大的 {len(data)} 個特徵（平均 |SHAP|）",
        xlabel="平均 |SHAP|（log-odds）",
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def plot_beeswarm(values: np.ndarray, X: pd.DataFrame, top: int = 12):
    import shap

    numeric = X.copy()
    for col in numeric.columns:
        if isinstance(numeric[col].dtype, pd.CategoricalDtype):
            numeric[col] = numeric[col].cat.codes
    cmap = LinearSegmentedColormap.from_list("value", ["#86b6ef", "#104281"])
    shap.summary_plot(values, numeric, max_display=top, show=False, cmap=cmap, plot_size=(7, 5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP 摘要圖（每點為一筆申請）", loc="left", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_dependence(values: np.ndarray, X: pd.DataFrame, table: pd.DataFrame, top: int = 3):
    numeric = [f for f in table["feature"] if not isinstance(X[f].dtype, pd.CategoricalDtype)][:top]
    fig, axes = plt.subplots(1, len(numeric), figsize=(3.4 * len(numeric), 3), squeeze=False)
    for ax, feature in zip(axes[0], numeric, strict=False):
        j = list(X.columns).index(feature)
        ax.scatter(X[feature], values[:, j], s=4, alpha=0.35, color=BLUE, rasterized=True)
        ax.axhline(0, color=INK_MUTED, lw=0.8)
        ax.set(title=feature, xlabel="特徵值", ylabel="SHAP")
    fig.tight_layout()
    return fig


def run(ctx: ExperimentContext) -> dict[str, pd.DataFrame]:
    use_report_style()
    section = ctx.section("explain")
    model_name = section["model"]
    if model_name not in TREE_MODELS:
        raise ValueError(f"TreeSHAP needs a tree model; got {model_name!r}")
    split = split_from_config(ctx.load(section.get("variant", "base")), ctx.config, "reproduction")
    X_train, y_train = split.xy("train")
    X_test, y_test = split.xy("test")
    params = runtime_params(ctx, model_name, default_params(ctx, model_name))
    model = build_model(model_name, params, random_state=ctx.seed).fit(X_train, y_train)
    s_train, s_test = predict_scores(model, X_train), predict_scores(model, X_test)
    threshold = threshold_at_fpr(y_test, s_test, ctx.target_fpr)
    flags_train, flags_test = s_train >= threshold, s_test >= threshold

    rng = np.random.default_rng(ctx.seed)
    n = min(int(section.get("shap_rows", 5000)), len(X_test))
    sample = np.sort(rng.choice(len(X_test), size=n, replace=False))
    outputs: dict[str, pd.DataFrame] = {}
    try:
        values = shap_values(model, X_test.iloc[sample])
        table = importance_table(values, X_test.iloc[sample])
        ctx.save_table(table, "e8_shap_importance")
        ctx.save_figure(plot_importance(table), "e8_shap_importance")
        ctx.save_figure(plot_beeswarm(values, X_test.iloc[sample]), "e8_shap_beeswarm")
        ctx.save_figure(plot_dependence(values, X_test.iloc[sample], table), "e8_shap_dependence")
        outputs["importance"] = table
    except ImportError:
        log.warning("shap not installed (pip install -e '.[explain]'); SHAP part skipped")

    fidelity_rows, rule_tables = [], []
    for depth in section.get("surrogate_depths", [3]):
        surrogate = fit_surrogate(
            X_train, flags_train, X_test, flags_test, max_depth=depth, seed=ctx.seed
        )
        fidelity_rows.append(
            {
                "depth": depth,
                "fidelity": surrogate.fidelity,
                "balanced_fidelity": surrogate.balanced_fidelity,
                "flag_recall": surrogate.flag_recall,
                "flag_precision": surrogate.flag_precision,
                "leaves": surrogate.tree.get_n_leaves(),
            }
        )
        rules = leaf_rules(surrogate, X_test, y_test, flags_test)
        rules.insert(0, "depth", depth)
        rule_tables.append(rules)
        (ctx.out_dir / f"surrogate_depth{depth}.txt").write_text(surrogate.text(), encoding="utf-8")
    fidelity = pd.DataFrame(fidelity_rows)
    rules = pd.concat(rule_tables, ignore_index=True)
    ctx.save_table(fidelity, "e8_surrogate_fidelity")
    ctx.save_table(rules, "e8_rules")
    ctx.save_table(
        rules.loc[rules["surrogate_flags"]].sort_values("fraud_rate", ascending=False).head(10),
        "e8_top_rules",
    )
    outputs |= {"fidelity": fidelity, "rules": rules}
    return outputs


__all__ = ["run", "shap_values"]
