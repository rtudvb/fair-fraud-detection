"""開戶風控決策支援系統 (E11).

Run locally:  streamlit run app/streamlit_app.py   (or `make app` from the repository root,
which also picks up the theme in .streamlit/config.toml)

Uses only aggregated score histograms (app/data/score_histogram.json, written by
`python -m frauddetect run app-data`). Without that file it shows clearly labelled demo data.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from frauddetect.config import load_yaml
from frauddetect.decision.three_tier import TierCosts
from frauddetect.simulator import GROUPS, ScoreHistogram

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app" / "data" / "score_histogram.json"
COSTS = ROOT / "configs" / "costs.yaml"
GROUP_NAMES = {"younger": "未滿 50 歲", "older": "50 歲以上"}
MODEL_NAMES = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}
TIERS = ("一級：正常開戶", "二級：加強驗證", "三級：人工審核")
TIER_COLORS = ("#C9D1DB", "#6F8DB5", "#1F3A5F")  # grey to navy, as in the report's figures
INK, MUTED, RULE, ACCENT, WARN = "#1F2328", "#5F6670", "#DADDE2", "#1F3A5F", "#9A3412"
FONT = '"Noto Sans TC", "Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", sans-serif'
MIN_FPR_RATIO = 0.8

CSS = f"""
<style>
html, body, .stApp, .stMarkdown, .stSlider, .stNumberInput {{ font-family: {FONT}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1180px; }}
h1 {{ font-size: 1.55rem !important; font-weight: 600 !important; padding-bottom: 0 !important; }}
section[data-testid="stSidebar"] h3 {{ font-size: 1rem; font-weight: 600; margin: .4rem 0 0; }}
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{ padding-top: 1.2rem; }}
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: .45rem; }}
section[data-testid="stSidebar"] label p {{ font-size: .85rem; }}
.meta {{ color: {MUTED}; font-size: .88rem; margin: .1rem 0 1.4rem; }}
.meta .demo {{ color: {WARN}; }}
.section {{ font-size: 1rem; font-weight: 600; color: {INK}; border-bottom: 1px solid {RULE};
           padding-bottom: .3rem; margin: 1.5rem 0 .7rem; }}
.kpi {{ border: 1px solid {RULE}; border-radius: 4px; padding: .85rem 1rem; background: #fff; }}
.kpi .label {{ color: {MUTED}; font-size: .85rem; }}
.kpi .value {{ color: {INK}; font-size: 1.85rem; font-weight: 600; margin: .15rem 0 .1rem;
              font-variant-numeric: tabular-nums; }}
.kpi .note {{ color: {MUTED}; font-size: .8rem; line-height: 1.45; min-height: 2.3em; }}
.kpi .warn {{ color: {WARN}; }}
table.report {{ width: 100%; border-collapse: collapse; font-size: .88rem;
               font-variant-numeric: tabular-nums; margin-bottom: 1rem; }}
table.report th, table.report td {{ padding: .32rem .45rem; border: none; }}
table.report thead th {{ border-top: 1.5px solid {INK}; border-bottom: 1px solid {INK};
                        font-weight: 600; }}
table.report tbody tr:last-child td {{ border-bottom: 1.5px solid {INK}; }}
table.report .num {{ text-align: right; }}
table.report .txt {{ text-align: left; }}
.caption {{ color: {MUTED}; font-size: .8rem; margin: -.2rem 0 .8rem; }}
.footnote {{ color: {MUTED}; font-size: .78rem; border-top: 1px solid {RULE};
            padding-top: .6rem; margin-top: 1.6rem; }}
</style>
"""


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


def defaults() -> tuple[dict, dict]:
    """Cost assumptions and staffing limits from configs/costs.yaml, as in the report."""
    config = load_yaml(COSTS) if COSTS.exists() else {}
    return config.get("tier_costs", {}), config.get("constraints", {})


def capacity(bar, limits: dict) -> tuple[float, float]:
    bar.markdown("### 人力上限")
    verify = bar.slider(
        "加強驗證上限（二級）",
        0.0,
        30.0,
        100 * float(limits.get("verify_capacity_rate", 0.10)),
        0.5,
        format="%.1f%%",
        help="驗證人力可處理的申請，占全部申請的比例。",
    )
    review = bar.slider(
        "人工審核上限（三級）",
        0.0,
        10.0,
        100 * float(limits.get("review_capacity_rate", 0.02)),
        0.5,
        format="%.1f%%",
        help="審核人員可處理的申請，占全部申請的比例。",
    )
    return verify / 100, review / 100


def sidebar(costs: dict) -> tuple[float, float, TierCosts]:
    bar = st.sidebar
    bar.markdown("### 門檻設定")
    escalate = (
        bar.slider(
            "升級比例（二級＋三級）",
            0.1,
            30.0,
            5.0,
            0.1,
            format="%.1f%%",
            help="風險分數最高、需要加強驗證或人工審核的申請，占全部申請的比例。",
        )
        / 100
    )
    review = (
        bar.slider(
            "其中轉人工審核（三級）",
            0.0,
            10.0,
            1.0,
            0.1,
            format="%.1f%%",
            help="分數最高、直接交由審核人員判斷的申請比例；其餘升級者進入二級加強驗證。",
        )
        / 100
    )
    review = min(review, escalate)
    bar.markdown("### 成本假設")
    values = {
        # Losses are entered in 萬元 (20 reads better than 200000) and converted back to 元.
        "loss_full": 10_000
        * bar.number_input(
            "詐騙帳戶完整功能的損失（萬元）",
            0,
            1_000,
            int(costs.get("loss_full", 200_000)) // 10_000,
            1,
        ),
        "loss_limited": 10_000
        * bar.number_input(
            "詐騙帳戶限額下的損失（萬元）",
            0,
            1_000,
            int(costs.get("loss_limited", 50_000)) // 10_000,
            1,
        ),
        "verify_catch": bar.slider(
            "加強驗證攔下詐騙的機率", 0.0, 1.0, float(costs.get("verify_catch", 0.5)), 0.05
        ),
        "review_catch": bar.slider(
            "人工審核攔下詐騙的機率", 0.0, 1.0, float(costs.get("review_catch", 0.9)), 0.05
        ),
        "cost_verify": bar.number_input(
            "每件加強驗證成本（元）", 0, 10_000, int(costs.get("cost_verify", 50)), 10
        ),
        "cost_review": bar.number_input(
            "每件人工審核成本（元）", 0, 10_000, int(costs.get("cost_review", 300)), 50
        ),
        "cost_friction": bar.number_input(
            "正常客戶被加強驗證的不便成本（元）",
            0,
            10_000,
            int(costs.get("cost_friction", 100)),
            10,
        ),
        "cost_attrition": bar.number_input(
            "正常客戶被轉人工的流失成本（元）",
            0,
            100_000,
            int(costs.get("cost_attrition", 2000)),
            100,
        ),
    }
    return escalate, review, TierCosts(**values)


def kpi(label: str, value: str, note: str) -> str:
    return (
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="note">{note}</div></div>'
    )


def table(headers: list[str], rows: list[list[str]], numeric: list[bool]) -> str:
    """A three-line table: text left, numbers right (the report's table style)."""
    cls = ["num" if n else "txt" for n in numeric]
    head = "".join(f'<th class="{c}">{h}</th>' for h, c in zip(headers, cls, strict=True))
    body = "".join(
        "<tr>" + "".join(f'<td class="{c}">{v}</td>' for v, c in zip(r, cls, strict=True)) + "</tr>"
        for r in rows
    )
    return f'<table class="report"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def tier_chart(result: dict) -> alt.Chart:
    """Share of all applications and of fraudulent ones in each tier, as 100% stacked bars."""
    frauds = np.array([result[f"fraud_tier{t}"] for t in (1, 2, 3)], dtype=float)
    groups = {
        "全部申請": [result[f"tier{t}_share"] for t in (1, 2, 3)],
        "詐騙申請": list(frauds / max(frauds.sum(), 1)),
    }
    rows = []
    for who, shares in groups.items():
        start = 0.0
        for tier, share in zip(TIERS, shares, strict=True):
            rows.append(
                {
                    "who": who,
                    "tier": tier,
                    "share": share,
                    "start": start,
                    "end": start + share,
                    "mid": start + share / 2,
                    "label": f"{share:.1%}" if share >= 0.045 else "",
                }
            )
            start += share
    data = pd.DataFrame(rows)
    y = alt.Y(
        "who:N",
        title=None,
        sort=list(groups),
        scale=alt.Scale(paddingInner=0.35, paddingOuter=0.25),
        axis=alt.Axis(ticks=False, domain=False, labelPadding=8),
    )
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "start:Q",
                title=None,
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", grid=False),
            ),
            x2="end:Q",
            y=y,
            color=alt.Color(
                "tier:N",
                title=None,
                scale=alt.Scale(domain=list(TIERS), range=list(TIER_COLORS)),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("who:N", title="對象"),
                alt.Tooltip("tier:N", title="等級"),
                alt.Tooltip("share:Q", title="比例", format=".1%"),
            ],
        )
    )
    labels = (
        alt.Chart(data)
        .mark_text(fontSize=12, fontWeight=600)
        .encode(
            x="mid:Q",
            y=y,
            text="label:N",
            color=alt.condition(alt.datum.tier == TIERS[0], alt.value(INK), alt.value("white")),
        )
    )
    return (bars + labels).properties(height={"step": 46})


def cost_curve(
    histogram: ScoreHistogram,
    escalate: float,
    review: float,
    costs: TierCosts,
    current: float,
    no_model: float,
    verify_cap: float,
) -> alt.Chart:
    """Expected cost per application as the escalation rate varies (review share held fixed).

    The grey band marks escalation rates whose tier 2 exceeds the verification capacity, so
    the falling curve is not read as "verify everyone".
    """
    rows = []
    for rate in np.round(np.arange(0.005, 0.3001, 0.005), 3):
        b1 = histogram.bin_for_alert_rate(rate)
        b2 = max(histogram.bin_for_alert_rate(min(review, rate)), b1)
        rows.append({"rate": rate, "cost": histogram.three_tier(b1, b2, costs)["cost_per_app"]})
    data = pd.DataFrame(rows)
    x = alt.X("rate:Q", title="升級比例（二級＋三級）", axis=alt.Axis(format="%", grid=False))
    y = alt.Y(
        "cost:Q",
        title="每件期望成本（元）",
        axis=alt.Axis(format=",.0f"),
        scale=alt.Scale(zero=False),
    )
    line = (
        alt.Chart(data)
        .mark_line(color=ACCENT, strokeWidth=2)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("rate:Q", title="升級比例", format=".1%"),
                alt.Tooltip("cost:Q", title="每件期望成本", format=",.0f"),
            ],
        )
    )
    here = pd.DataFrame({"rate": [escalate], "cost": [current]})
    point = (
        alt.Chart(here)
        .mark_point(filled=True, size=90, color=ACCENT)
        .encode(x="rate:Q", y="cost:Q")
    )
    tag = halo_text(here, "目前設定", dx=8, dy=-12, x="rate:Q", y="cost:Q")
    base = pd.DataFrame({"rate": [data["rate"].max()], "cost": [no_model], "text": ["不使用模型"]})
    rule = alt.Chart(base).mark_rule(strokeDash=[4, 4], color=MUTED).encode(y="cost:Q")
    rule_tag = (
        alt.Chart(base)
        .mark_text(align="right", dy=-7, fontSize=11, color=MUTED)
        .encode(x="rate:Q", y="cost:Q", text="text:N")
    )
    layers = [rule, rule_tag, line, point, tag]
    limit = verify_cap + review  # tier 2 = escalated - reviewed
    if limit < data["rate"].max():
        band = pd.DataFrame(
            {"rate": [limit], "rate_end": [data["rate"].max()], "text": ["超出加強驗證人力"]}
        )
        shade = alt.Chart(band).mark_rect(color="#EEF0F2").encode(x=x, x2="rate_end:Q")
        edge = alt.Chart(band).mark_rule(color=MUTED, strokeDash=[2, 3]).encode(x=x)
        note = halo_text(band, "超出加強驗證人力", dx=6, dy=0, x=x, y=alt.value(44), color=MUTED)
        layers = [shade, edge, note, *layers]
    return alt.layer(*layers).properties(height=260)


def halo_text(data: pd.DataFrame, text: str, *, dx: int, dy: int, x, y, color: str = INK):
    """A left-aligned label drawn over a white outline of itself."""
    style = {"align": "left", "dx": dx, "dy": dy, "fontSize": 12}
    under = alt.Chart(data).mark_text(stroke="white", strokeWidth=4, **style)
    over = alt.Chart(data).mark_text(color=color, **style)
    encoding = {"x": x, "y": y, "text": alt.value(text)}
    return alt.layer(under.encode(**encoding), over.encode(**encoding))


def configure(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure(font=FONT)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            labelFontSize=11,
            titleFontSize=12,
            domainColor=RULE,
            tickColor=RULE,
            gridColor="#EEF0F2",
        )
        .configure_legend(labelColor=INK, labelFontSize=12, symbolType="square")
    )


def main() -> None:
    st.set_page_config(page_title="開戶風控決策支援系統", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    histogram, demo = load_histogram()
    cost_defaults, limits = defaults()
    escalate, review, costs = sidebar(cost_defaults)
    verify_cap, review_cap = capacity(st.sidebar, limits)

    st.title("開戶風控決策支援系統")
    source = (
        '<span class="demo">示範資料（隨機產生，不是研究結果）</span>'
        if demo
        else f"資料：{histogram.period}，共 {histogram.total():,} 件申請"
    )
    st.markdown(
        f'<div class="meta">{source}｜模型：{MODEL_NAMES.get(histogram.model, histogram.model)}'
        "｜僅使用彙總後的分數分布</div>",
        unsafe_allow_html=True,
    )

    t1 = histogram.bin_for_alert_rate(escalate)
    t2 = max(histogram.bin_for_alert_rate(review), t1)
    escalated, reviewed = histogram.rates(t1), histogram.rates(t2)
    result = histogram.three_tier(t1, t2, costs)
    no_model = histogram.three_tier(histogram.n_bins, histogram.n_bins, costs)
    frauds = histogram.total(1)
    ratio = escalated["fpr_ratio"]
    saving = (
        1 - result["cost_per_app"] / no_model["cost_per_app"] if no_model["cost_per_app"] else 0
    )
    fairness_note = (
        f'<span class="warn">未達 {MIN_FPR_RATIO} 參考值：年長的正常客戶較常被升級</span>'
        if ratio < MIN_FPR_RATIO
        else f"達到 {MIN_FPR_RATIO} 參考值"
    )
    cards = [
        kpi(
            "詐騙預期攔截率",
            f"{result['fraud_caught'] / max(frauds, 1):.1%}",
            f"升級涵蓋 {escalated['tpr']:.1%}（{round(escalated['tpr'] * frauds):,} 件），"
            f"預期攔下 {round(result['fraud_caught']):,} 件，共 {frauds:,} 件詐騙申請",
        ),
        kpi("正常客戶受影響比例", f"{escalated['fpr']:.2%}", "需加強驗證或人工審核的正常申請"),
        kpi("年齡公平性（FPR 比）", f"{ratio:.2f}", fairness_note),
        kpi(
            "每件申請期望成本",
            f"{result['cost_per_app']:,.0f} 元",
            f"不使用模型：{no_model['cost_per_app']:,.0f} 元（節省 {saving:.0%}）",
        ),
    ]
    for column, card in zip(st.columns(4), cards, strict=True):
        column.markdown(card, unsafe_allow_html=True)

    st.markdown('<div class="section">分流結果</div>', unsafe_allow_html=True)
    st.altair_chart(configure(tier_chart(result)), use_container_width=True)
    st.markdown(
        '<div class="meta">門檻取在彙總分數區間的切點上，實際比例為不超過設定值、最接近的切點，'
        "因此可能略低於左側的設定。</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<div class="section">升級比例與期望成本</div>', unsafe_allow_html=True)
        st.altair_chart(
            configure(
                cost_curve(
                    histogram,
                    escalate,
                    review,
                    costs,
                    result["cost_per_app"],
                    no_model["cost_per_app"],
                    verify_cap,
                )
            ),
            use_container_width=True,
        )

        def usage(name: str, share: float, cap: float) -> str:
            text = f"{name}占 {share:.1%}（上限 {cap:.1%}）"
            return f'<span class="warn">{text}，超過上限</span>' if share > cap + 1e-9 else text

        st.markdown(
            '<div class="caption">目前設定：'
            f"{usage('二級（加強驗證）', result['tier2_share'], verify_cap)}；"
            f"{usage('三級（人工審核）', result['tier3_share'], review_cap)}。"
            "曲線固定目前的人工審核比例；虛線為不使用模型時的期望成本。</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="section">成本組成（元／件）</div>', unsafe_allow_html=True)
        keys = ("fraud_loss_per_app", "operations_per_app", "customer_per_app", "cost_per_app")
        names = ("詐騙損失", "驗證與審核作業", "正常客戶的不便與流失", "合計")
        st.markdown(
            table(
                ["項目", "目前設定", "不使用模型"],
                [
                    [n, f"{result[k]:,.0f}", f"{no_model[k]:,.0f}"]
                    for n, k in zip(names, keys, strict=True)
                ],
                [False, True, True],
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section">各年齡組</div>', unsafe_allow_html=True)
        metrics = (  # one row per measure, one column per age group
            ("正常客戶被升級", "fpr", escalated, ".2%"),
            ("正常客戶被轉人工", "fpr", reviewed, ".2%"),
            ("詐騙被升級", "tpr", escalated, ".1%"),
        )
        st.markdown(
            table(
                ["指標", *(GROUP_NAMES[g] for g in GROUPS)],
                [
                    [name, *(format(rates[f"{key}_{g}"], fmt) for g in GROUPS)]
                    for name, key, rates, fmt in metrics
                ],
                [False, True, True],
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="footnote">資料：BAF（Jesus et al., NeurIPS 2022；CC BY-NC-ND 4.0）之彙總統計。'
        "成本與帳戶限額為情境假設，不代表任何銀行的實際數字。</div>",
        unsafe_allow_html=True,
    )


main()
