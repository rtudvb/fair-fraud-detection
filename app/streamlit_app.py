"""風控門檻模擬器 — Catch Fraud, Not Customers (E11).

Run locally:  streamlit run app/streamlit_app.py

Uses only aggregated score histograms (app/data/score_histogram.json, written by
`python -m frauddetect run app-data`). Without that file it shows clearly labelled demo data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from frauddetect.config import load_yaml
from frauddetect.decision.three_tier import TierCosts
from frauddetect.simulator import GROUPS, ScoreHistogram
from frauddetect.viz import TIER_COLORS, TIER_LABELS, use_report_style

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data" / "score_histogram.json"
COSTS = ROOT / "configs" / "costs.yaml"
GROUP_NAMES = {"younger": "未滿 50 歲", "older": "50 歲以上"}


@st.cache_data
def load_histogram() -> tuple[ScoreHistogram, bool]:
    if DATA.exists():
        return ScoreHistogram.from_json(DATA), False
    rng = np.random.default_rng(0)  # demo data: random scores, NOT results
    n = 200_000
    groups = np.where(rng.random(n) < 0.23, "older", "younger")
    y = (rng.random(n) < np.where(groups == "older", 0.018, 0.009)).astype(int)
    scores = 1 / (1 + np.exp(-(rng.normal(size=n) + 2.2 * y + 0.5 * (groups == "older") - 4)))
    return ScoreHistogram.from_scores(y, scores, groups, model="示範模型", period="示範資料"), True


def default_costs() -> dict:
    return load_yaml(COSTS)["tier_costs"] if COSTS.exists() else {}


def sidebar(costs: dict) -> tuple[float, float, TierCosts]:
    st.sidebar.header("門檻設定")
    escalate = st.sidebar.slider("升級比例（二級＋三級，占全部申請）", 0.1, 30.0, 5.0, 0.1) / 100
    review = st.sidebar.slider("其中轉人工審核（三級）", 0.0, 10.0, 1.0, 0.1) / 100
    review = min(review, escalate)
    st.sidebar.header("成本假設（元）")
    values = {
        "loss_full": st.sidebar.number_input(
            "詐騙帳戶完整功能的損失", 0, 10_000_000, int(costs.get("loss_full", 200_000)), 10_000
        ),
        "loss_limited": st.sidebar.number_input(
            "詐騙帳戶限額下的損失", 0, 10_000_000, int(costs.get("loss_limited", 50_000)), 10_000
        ),
        "verify_catch": st.sidebar.slider(
            "加強驗證攔下詐騙的機率", 0.0, 1.0, float(costs.get("verify_catch", 0.5)), 0.05
        ),
        "review_catch": st.sidebar.slider(
            "人工審核攔下詐騙的機率", 0.0, 1.0, float(costs.get("review_catch", 0.9)), 0.05
        ),
        "cost_verify": st.sidebar.number_input(
            "每件加強驗證成本", 0, 10_000, int(costs.get("cost_verify", 50)), 10
        ),
        "cost_review": st.sidebar.number_input(
            "每件人工審核成本", 0, 10_000, int(costs.get("cost_review", 300)), 50
        ),
        "cost_friction": st.sidebar.number_input(
            "好客戶被加強驗證的不便成本", 0, 10_000, int(costs.get("cost_friction", 100)), 10
        ),
        "cost_attrition": st.sidebar.number_input(
            "好客戶被轉人工的流失成本", 0, 100_000, int(costs.get("cost_attrition", 2000)), 100
        ),
    }
    return escalate, review, TierCosts(**values)


def tier_chart(result: dict):
    use_report_style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    shares = np.array([result[f"tier{t}_share"] for t in (1, 2, 3)]) * 100
    frauds = np.array([result[f"fraud_tier{t}"] for t in (1, 2, 3)], dtype=float)
    fraud_shares = frauds / max(frauds.sum(), 1) * 100
    colors = [TIER_COLORS[t] for t in (1, 2, 3)]
    labels = [TIER_LABELS[t].split("：")[0] for t in (1, 2, 3)]
    for ax, values, title in zip(
        axes, (shares, fraud_shares), ("全部申請", "詐騙申請"), strict=True
    ):
        ax.bar(labels, values, color=colors, width=0.6)
        ax.set(title=f"{title}分到各級的比例", ylabel="%")
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="風控門檻模擬器", page_icon="🛡️", layout="wide")
    st.title("風控門檻模擬器")
    st.caption("Catch Fraud, Not Customers — 調整門檻，即時看偵測率、年齡公平性與期望成本。")
    histogram, demo = load_histogram()
    if demo:
        st.warning("目前顯示的是隨機產生的示範資料，不是研究結果。")
    else:
        st.info(f"模型：{histogram.model}；期間：{histogram.period}（僅使用彙總後的分數分布）")

    escalate, review, costs = sidebar(default_costs())
    t1 = histogram.bin_for_alert_rate(escalate)
    t2 = max(histogram.bin_for_alert_rate(review), t1)
    escalated = histogram.rates(t1)
    reviewed = histogram.rates(t2)
    result = histogram.three_tier(t1, t2, costs)
    no_model = histogram.three_tier(histogram.n_bins, histogram.n_bins, costs)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("攔到的詐騙（二級＋三級）", f"{escalated['tpr']:.1%}")
    c2.metric("好客戶被升級的比例", f"{escalated['fpr']:.2%}")
    ratio = escalated["fpr_ratio"]
    c3.metric("年齡公平性（FPR 比）", f"{ratio:.2f}")
    c3.caption("✅ 達 0.8 標準" if ratio >= 0.8 else "⚠️ 未達 0.8 標準：年長的好客戶較常被升級")
    saving = (
        1 - result["cost_per_app"] / no_model["cost_per_app"] if no_model["cost_per_app"] else 0
    )
    c4.metric("每件申請期望成本", f"{result['cost_per_app']:,.0f} 元", f"比不用模型省 {saving:.0%}")

    st.pyplot(tier_chart(result))

    left, right = st.columns(2)
    with left:
        st.subheader("各年齡組")
        st.table(
            {
                "組別": [GROUP_NAMES[g] for g in GROUPS],
                "好客戶被升級": [f"{escalated[f'fpr_{g}']:.2%}" for g in GROUPS],
                "好客戶被轉人工": [f"{reviewed[f'fpr_{g}']:.2%}" for g in GROUPS],
                "詐騙被攔下（二級以上）": [f"{escalated[f'tpr_{g}']:.1%}" for g in GROUPS],
            }
        )
    with right:
        st.subheader("成本組成（元／件）")
        st.table(
            {
                "項目": ["詐騙損失", "審核與驗證", "好客戶不便與流失", "合計"],
                "目前設定": [
                    f"{result[k]:,.0f}"
                    for k in (
                        "fraud_loss_per_app",
                        "operations_per_app",
                        "customer_per_app",
                        "cost_per_app",
                    )
                ],
                "不用模型": [
                    f"{no_model[k]:,.0f}"
                    for k in (
                        "fraud_loss_per_app",
                        "operations_per_app",
                        "customer_per_app",
                        "cost_per_app",
                    )
                ],
            }
        )
    st.caption(
        "資料：BAF（Jesus et al., NeurIPS 2022, CC BY-NC-ND 4.0）之彙總統計。"
        "成本與帳戶限額為情境假設，不代表任何銀行的實際數字。"
    )


main()
