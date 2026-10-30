"""Draw the two temporal split protocols of chapter 3 (圖 3-2).

Run: python reports/paper/diagrams/data_split.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

from frauddetect.viz import save_figure, use_report_style
from frauddetect.viz.style import BLUE, INK, INK_SECONDARY, ORANGE

OUT = Path(__file__).resolve().parents[1] / "figures"
TRAIN, VALID, TEST = "#86b6ef", ORANGE, BLUE

PROTOCOLS = [
    (
        "最終評估（final）",
        "模型比較、公平性量測、重現（RQ1–RQ3、RQ5）",
        [TRAIN] * 6 + [TEST] * 2,
    ),
    (
        "開發（dev）",
        "調參、門檻與決策政策選擇（RQ3 處理、RQ4）",
        [TRAIN] * 5 + [VALID] + [TEST] * 2,
    ),
]


def draw():
    use_report_style()
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for row, (name, purpose, colors) in enumerate(PROTOCOLS):
        y = 1.6 - row * 1.25
        ax.text(
            -0.25,
            y + 0.42,
            name,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
        ax.text(-0.25, y + 0.05, purpose, ha="right", va="center", fontsize=8, color=INK_SECONDARY)
        for month, color in enumerate(colors):
            ax.add_patch(Rectangle((month, y), 0.94, 0.62, facecolor=color, edgecolor="white"))
    for month in range(8):
        ax.text(month + 0.47, -0.05, f"{month}", ha="center", va="top", fontsize=9, color=INK)
    ax.text(4.0, -0.42, "月份（month）", ha="center", va="top", fontsize=9, color=INK_SECONDARY)
    ax.legend(
        handles=[
            Patch(facecolor=TRAIN, label="訓練"),
            Patch(facecolor=VALID, label="驗證"),
            Patch(facecolor=TEST, label="測試"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.62, 1.18),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    ax.set_xlim(-5.4, 8.1)
    ax.set_ylim(-0.75, 2.45)
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print(save_figure(draw(), "fig3_data_split", OUT))
