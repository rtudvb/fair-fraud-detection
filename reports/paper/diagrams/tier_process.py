"""Draw the proposed three-tier account-opening review process (圖 5-1), formal style.

Run: python reports/paper/diagrams/tier_process.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from research_framework import INK, arrow, box, use_serif

from frauddetect.viz import save_figure

OUT = Path(__file__).resolve().parents[1] / "figures"


def label(ax, x, y, text, ha="center"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=9.5,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
    )


def draw():
    use_serif()
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12.6)
    ax.axis("off")

    w, h = 3.4, 1.1
    box(ax, 8.3, 11.0, w, h, "線上開戶申請")
    box(ax, 8.3, 8.9, w, h, "模型風險評分")
    arrow(ax, (10.0, 11.0), (10.0, 10.0))

    # Fan out into the three tiers.
    xs = [3.0, 10.0, 17.0]
    split_y = 7.9
    ax.plot([10.0, 10.0], [8.9, split_y], color=INK, lw=0.9)
    ax.plot([xs[0], xs[-1]], [split_y, split_y], color=INK, lw=0.9)
    tiers = ["一級：正常開戶", "二級：限額開戶", "三級：人工審核"]
    conditions = ["分數 < $t_1$", "$t_1$ ≤ 分數 < $t_2$", "分數 ≥ $t_2$"]
    for x, tier, cond in zip(xs, tiers, conditions, strict=True):
        arrow(ax, (x, split_y), (x, 6.6))
        label(ax, x, 7.25, cond)
        box(ax, x - w / 2, 5.5, w, h, tier)

    # Tier 2: enhanced verification lifts the limit; failures go to manual review.
    box(ax, 10.0 - w / 2, 3.2, w, h, "加強驗證\n（視訊或電信）")
    arrow(ax, (10.0, 5.5), (10.0, 4.3))
    box(ax, 10.0 - w / 2, 0.6, w, h, "解除限額")
    arrow(ax, (10.0, 3.2), (10.0, 1.7))
    label(ax, 10.0, 2.45, "通過")
    arrow(ax, (10.0 + w / 2, 3.75), (17.0 - w / 2, 3.75))
    label(ax, 13.5, 3.75, "未通過")

    # Tier 3: manual review decides.
    box(ax, 17.0 - w / 2, 3.2, w, h, "審核人員判斷")
    arrow(ax, (17.0, 5.5), (17.0, 4.3))
    box(ax, 17.0 - w / 2, 0.6, w, h, "核准或拒絕")
    arrow(ax, (17.0, 3.2), (17.0, 1.7))

    # Tier 1 opens with the full functionality of its account type.
    box(ax, 3.0 - w / 2, 3.2, w, h, "完整功能帳戶")
    arrow(ax, (3.0, 5.5), (3.0, 4.3))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print(save_figure(draw(), "fig5_tier_process", OUT))
