"""E9 — post-deployment monitoring and the model card (report §5).

* Feature drift: PSI of every feature, training months 0-5 vs each test month.
* Month-by-month tracking: the threshold is fixed on validation month 5 (as at deployment)
  and the alert rate, FPR, recall and age-fairness of the ``dev`` model are tracked over
  months 5-7, which is what a bank would watch after go-live.
* Model card: a Markdown summary assembled from whatever results exist.
"""

from __future__ import annotations

import logging
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from frauddetect.data.schema import LABEL, MONTH
from frauddetect.data.split import split_from_config
from frauddetect.evaluation.fairness import age_groups, fpr_ratio
from frauddetect.evaluation.metrics import threshold_at_fpr
from frauddetect.experiments.context import ExperimentContext
from frauddetect.experiments.scoring import get_scores
from frauddetect.monitoring.psi import psi_table
from frauddetect.viz import use_report_style
from frauddetect.viz.style import BLUE, INK_MUTED, ORANGE

log = logging.getLogger(__name__)


def drift_table(ctx: ExperimentContext, variant: str) -> pd.DataFrame:
    split = split_from_config(ctx.load(variant), ctx.config, "reproduction")
    reference = split.train.drop(columns=[LABEL, MONTH])
    merged = None
    for month in sorted(split.test[MONTH].unique()):
        current = split.test.loc[split.test[MONTH] == month].drop(columns=[LABEL, MONTH])
        table = psi_table(reference, current)[["feature", "psi"]]
        table = table.rename(columns={"psi": f"psi_month{month}"})
        merged = table if merged is None else merged.merge(table, on="feature")
    psi_cols = [c for c in merged.columns if c.startswith("psi_")]
    merged["psi_max"] = merged[psi_cols].max(axis=1)
    merged["status"] = pd.cut(
        merged["psi_max"], [-np.inf, 0.1, 0.25, np.inf], labels=["穩定", "中度變化", "顯著變化"]
    ).astype(str)
    return merged.sort_values("psi_max", ascending=False, ignore_index=True)


def monthly_tracking(ctx: ExperimentContext, model: str, variant: str) -> pd.DataFrame:
    run = get_scores(ctx, model, protocol="dev", variant=variant)
    scores = run.scores
    valid = scores.loc[scores["part"] == "valid"]
    threshold = threshold_at_fpr(valid["y"], valid["score"], ctx.target_fpr)
    cutoff = int(ctx.config["fairness"]["age_cutoff"])
    rows = []
    for month, part in scores.groupby("month"):
        y = part["y"].to_numpy().astype(bool)
        flagged = part["score"].to_numpy() >= threshold
        rows.append(
            {
                "month": int(month),
                "role": "驗證（設定門檻）" if month in set(valid["month"]) else "上線後",
                "applications": len(part),
                "fraud_rate": y.mean(),
                "alert_rate": flagged.mean(),
                "fpr": flagged[~y].mean(),
                "recall": flagged[y].mean() if y.any() else np.nan,
                "fpr_ratio": fpr_ratio(y, flagged, age_groups(part["age"], cutoff)),
            }
        )
    return pd.DataFrame(rows)


def plot_monthly(table: pd.DataFrame, target_fpr: float):
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 3.2))
    months = table["month"]
    left.plot(months, table["recall"], marker="o", color=BLUE, label="偵測率")
    left.plot(months, table["fpr"], marker="s", color=ORANGE, label="誤判率")
    left.axhline(target_fpr, color=INK_MUTED, lw=0.8, ls="--")
    left.set(title="門檻固定後的逐月表現", xlabel="月份", ylim=(0, None))
    left.legend(fontsize=8)
    right.plot(months, table["fpr_ratio"], marker="o", color=BLUE)
    right.axhline(0.8, color=INK_MUTED, lw=0.8, ls="--")
    right.set(title="逐月年齡公平性", xlabel="月份", ylabel="FPR 比", ylim=(0, 1.02))
    for ax in (left, right):
        ax.set_xticks(months)
    fig.tight_layout()
    return fig


def model_card(
    ctx: ExperimentContext, model: str, monthly: pd.DataFrame, drift: pd.DataFrame
) -> str:
    """Markdown model card filled from available results (missing parts are marked)."""

    def from_table(experiment: str, name: str) -> pd.DataFrame | None:
        path = ctx.results_root / experiment / f"{name}.csv"
        return pd.read_csv(path) if path.exists() else None

    perf = from_table("benchmark", "e2_default")
    fair = from_table("fairness", "e5_group_rates")
    lines = [
        f"# 模型說明書（Model Card）：{model}",
        "",
        f"產生日期：{date.today().isoformat()}　資料：BAF Base（Jesus et al., 2022）",
        "",
        "## 用途",
        "- 預期用途：銀行線上開戶申請的詐騙風險評分，"
        "作為三級分流（正常／限額＋加強驗證／人工審核）的依據。",
        "- 不適用：單獨作為拒絕開戶的依據；BAF 以外、未經驗證的資料。",
        "",
        "## 資料與訓練",
        "- 訓練：第 0–5 月；測試：第 6–7 月（時間切分，避免資訊洩漏）。",
        "- 資料為以真實銀行資料生成的合成資料（CC BY-NC-ND 4.0），不含可識別個人資訊。",
        "",
        "## 效能（測試期，誤判率 5%）",
    ]
    if perf is not None and model in set(perf["model"]):
        row = perf.set_index("model").loc[model]
        lines += [
            f"- 偵測率 TPR@5%FPR：{row['tpr_at_fpr']:.3f}"
            f"（95% CI {row['tpr_ci_low']:.3f}–{row['tpr_ci_high']:.3f}）",
            f"- ROC-AUC：{row['roc_auc']:.3f}；PR-AUC：{row['pr_auc']:.3f}",
        ]
    else:
        lines.append("- （待 E2 結果）")
    lines += ["", "## 公平性（年齡 ≥ 50 vs < 50）"]
    if fair is not None and model in set(fair["model"]):
        row = fair.set_index("model").loc[model]
        lines.append(
            f"- 誤判率：未滿 50 歲 {row['fpr_younger']:.3f}、50 歲以上 {row['fpr_older']:.3f}；"
            f"FPR 比 {row['fpr_ratio']:.2f}"
        )
    else:
        lines.append("- （待 E5 結果）")
    unstable = drift.loc[drift["status"] != "穩定", "feature"].tolist()
    lines += [
        "",
        "## 上線監控",
        "- 逐月偵測率："
        + "、".join(
            f"第 {m} 月 {r:.3f}" for m, r in zip(monthly["month"], monthly["recall"], strict=True)
        ),
        f"- 資料漂移（PSI ≥ 0.1）的特徵：{', '.join(unstable) if unstable else '無'}",
        "- 建議：每月重算 PSI 與分組誤判率；FPR 比低於 0.8 或偵測率下降超過 5 個百分點時重新訓練。",
        "",
        "## 限制",
        "- 資料非台灣銀行資料；三級分流的成本為情境假設。",
        "- 只涵蓋開戶階段；既有帳戶轉為人頭帳戶需搭配交易監控。",
    ]
    return "\n".join(lines) + "\n"


def run(ctx: ExperimentContext) -> dict[str, pd.DataFrame]:
    use_report_style()
    section = ctx.section("monitoring")
    variant, model = section.get("variant", "base"), section["model"]
    drift = drift_table(ctx, variant)
    ctx.save_table(drift, "e9_psi")
    monthly = monthly_tracking(ctx, model, variant)
    ctx.save_table(monthly, "e9_monthly")
    ctx.save_figure(plot_monthly(monthly, ctx.target_fpr), "e9_monthly")
    card = model_card(ctx, model, monthly, drift)
    (ctx.out_dir / "model_card.md").write_text(card, encoding="utf-8")
    (ctx.tables_dir / "e9_model_card.md").write_text(card, encoding="utf-8")
    return {"drift": drift, "monthly": monthly}
