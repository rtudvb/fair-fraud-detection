"""E10 — the small-bank dilemma (RQ6, report §4.7).

The training months are split across five simulated banks (unequal sizes, Dirichlet-skewed
customer mix on ``partition_feature``); the test months are split the same way, so each bank
is evaluated on its own future customers. Methods:

* local models trained only on the bank's own data (boosted trees, foundation model, MLP);
* FedAvg MLP: one model trained jointly without sharing raw data;
* pooled training (all data in one place) as the upper reference.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frauddetect.data.schema import LABEL, MONTH
from frauddetect.data.split import split_from_config, stratified_subsample
from frauddetect.evaluation.metrics import tpr_at_fpr
from frauddetect.evaluation.stats import bootstrap_ci
from frauddetect.experiments.context import ExperimentContext
from frauddetect.experiments.scoring import default_params, evaluate_scores, runtime_params
from frauddetect.federated import (
    FederatedPreprocessor,
    TrainConfig,
    assign_banks,
    bank_probabilities,
    predict,
    train_fedavg,
    train_single,
)
from frauddetect.models.registry import build_model, predict_scores
from frauddetect.viz import use_report_style
from frauddetect.viz.style import AQUA, BLUE, INK_MUTED, ORANGE

log = logging.getLogger(__name__)

METHOD_STYLE = {  # color = where training happens, marker/line = model
    "local_lightgbm": (BLUE, "o", "-", "各行自建：LightGBM"),
    "local_tabpfn": (BLUE, "s", "--", "各行自建：TabPFN-3.5"),
    "local_mlp": (BLUE, "^", ":", "各行自建：MLP"),
    "fedavg_mlp": (ORANGE, "D", "-", "聯合學習：MLP（FedAvg）"),
    "pooled_mlp": (AQUA, "^", ":", "資料集中：MLP（參考）"),
    "pooled_lightgbm": (AQUA, "o", "-", "資料集中：LightGBM（參考）"),
}


def run(ctx: ExperimentContext) -> pd.DataFrame:
    use_report_style()
    section = ctx.section("small_bank")
    cutoff = int(ctx.config["fairness"]["age_cutoff"])
    split = split_from_config(ctx.load(section.get("variant", "base")), ctx.config, "reproduction")
    feature = section["partition_feature"]
    levels = sorted(split.train[feature].dropna().astype(str).unique())
    probs = bank_probabilities(levels, section["bank_shares"], section["dirichlet_alpha"], ctx.seed)
    train_bank = assign_banks(split.train[feature], probs, ctx.seed)
    test_bank = assign_banks(split.test[feature], probs, ctx.seed + 1)
    n_banks = len(section["bank_shares"])
    # Name banks by size: A is the largest, the last letter the smallest.
    rank = np.argsort(np.argsort(-np.bincount(train_bank, minlength=n_banks)))
    train_bank, test_bank = rank[train_bank], rank[test_bank]
    banks = [
        (split.train.loc[train_bank == b], split.test.loc[test_bank == b]) for b in range(n_banks)
    ]
    ctx.save_table(_bank_profile(banks, feature), "e10_banks")

    scores: dict[str, list[np.ndarray]] = {}
    for model in section.get("local_models", ["lightgbm"]):
        try:
            scores[f"local_{model}"] = [
                _fit_tabular(ctx, model, train, test) for train, test in banks
            ]
        except Exception as exc:
            log.error("local %s failed: %s", model, exc)
    pooled_train = pd.concat([train for train, _ in banks])
    pooled = build_model(
        "lightgbm",
        runtime_params(ctx, "lightgbm", default_params(ctx, "lightgbm")),
        random_state=ctx.seed,
    ).fit(pooled_train.drop(columns=[LABEL, MONTH]), pooled_train[LABEL])
    scores["pooled_lightgbm"] = [
        predict_scores(pooled, test.drop(columns=[LABEL, MONTH])) for _, test in banks
    ]
    try:
        scores |= _neural_methods(ctx, section, banks)
    except ImportError:
        log.warning("PyTorch not installed; MLP / FedAvg methods skipped")

    # Small banks have few test frauds, so every recall gets a bootstrap interval.
    n_boot = int(section.get("n_boot", 1000))
    rows = []
    for method, per_bank in scores.items():
        for b, ((train, test), s) in enumerate(zip(banks, per_bank, strict=True)):
            y = test[LABEL].to_numpy()
            metrics = evaluate_scores(
                y, s, test["customer_age"].to_numpy(), target_fpr=ctx.target_fpr, cutoff=cutoff
            )
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
                    "bank": f"銀行 {chr(65 + b)}",
                    "bank_index": b,
                    "method": method,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_test_frauds": int(y.sum()),
                    **metrics,
                }
            )
    results = pd.DataFrame(rows)
    ctx.save_table(results, "e10_small_bank")
    wide = results.pivot_table(index="method", columns="bank", values="tpr_at_fpr").reset_index()
    ctx.save_table(wide, "e10_small_bank_wide")
    ctx.save_figure(plot_banks(results), "e10_small_bank")
    return results


def _bank_profile(banks, feature: str) -> pd.DataFrame:
    rows = []
    for b, (train, test) in enumerate(banks):
        mix = train[feature].astype(str).value_counts(normalize=True)
        rows.append(
            {
                "bank": f"銀行 {chr(65 + b)}",
                "n_train": len(train),
                "n_test": len(test),
                "frauds_train": int(train[LABEL].sum()),
                "fraud_rate_train": train[LABEL].mean(),
                f"top_{feature}": f"{mix.index[0]} ({mix.iloc[0]:.0%})",
            }
        )
    return pd.DataFrame(rows)


def _fit_tabular(ctx, model: str, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    train = stratified_subsample(train, ctx.max_train_rows(model), seed=ctx.seed)
    estimator = build_model(
        model, runtime_params(ctx, model, default_params(ctx, model)), random_state=ctx.seed
    ).fit(train.drop(columns=[LABEL, MONTH]), train[LABEL])
    batch = ctx.config["experiments"]["runtime"].get("predict_batch_size")
    return predict_scores(estimator, test.drop(columns=[LABEL, MONTH]), batch_size=batch)


def _neural_methods(ctx, section, banks) -> dict[str, list[np.ndarray]]:
    fed = section.get("fedavg", {})
    features = [train.drop(columns=[LABEL, MONTH]) for train, _ in banks]
    prep = FederatedPreprocessor.from_clients(features)
    clients = [
        (prep.transform(f), train[LABEL].to_numpy())
        for f, (train, _) in zip(features, banks, strict=True)
    ]
    tests = [prep.transform(test.drop(columns=[LABEL, MONTH])) for _, test in banks]
    positives = sum(int(y.sum()) for _, y in clients)
    total = sum(len(y) for _, y in clients)
    cfg = TrainConfig(
        hidden=tuple(fed.get("hidden", (128, 64))),
        batch_size=int(fed.get("batch_size", 1024)),
        lr=float(fed.get("lr", 1e-3)),
        local_epochs=int(fed.get("local_epochs", 1)),
        rounds=int(fed.get("rounds", 30)),
        pos_weight=(total - positives) / max(positives, 1),
        device=ctx.device,
        seed=ctx.seed,
    )
    out: dict[str, list[np.ndarray]] = {}
    out["local_mlp"] = [
        predict(train_single(X, y, cfg), X_test, cfg.device)
        for (X, y), X_test in zip(clients, tests, strict=True)
    ]
    federated = train_fedavg(clients, cfg)
    out["fedavg_mlp"] = [predict(federated, X_test, cfg.device) for X_test in tests]
    pooled = train_single(
        np.vstack([X for X, _ in clients]), np.concatenate([y for _, y in clients]), cfg
    )
    out["pooled_mlp"] = [predict(pooled, X_test, cfg.device) for X_test in tests]
    return out


def plot_banks(results: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    banks = results.drop_duplicates("bank_index").sort_values("bank_index")
    sizes = dict(zip(banks["bank_index"], banks["n_train"], strict=True))
    for method, part in results.groupby("method"):
        color, marker, line, label = METHOD_STYLE.get(method, (INK_MUTED, "o", "-", method))
        part = part.sort_values("bank_index")
        ax.plot(
            part["bank_index"], part["tpr_at_fpr"], color=color, marker=marker, ls=line, label=label
        )
    ax.set_xticks(list(sizes), [f"{chr(65 + b)}\n{n:,} 筆" for b, n in sizes.items()], fontsize=8)
    ax.set(
        title="資料多寡不同的銀行：各自建模 vs 聯合學習",
        xlabel="銀行（訓練資料量）",
        ylabel="偵測率（TPR@5%FPR）",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return fig
