"""Draw the two temporal split protocols of chapter 3 (圖 3-5) in the formal style of 圖 3-1.

Black outlines on white; training, validation and test are told apart by fill (white, light
grey, hatching) and named inside the bars, so the figure needs no colour and no legend.

Run: python reports/paper/diagrams/data_split.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from research_framework import INK, use_serif

from frauddetect.viz import save_figure

OUT = Path(__file__).resolve().parents[1] / "figures"
STYLE = {  # fill and hatch of each role
    "訓練": {"facecolor": "white", "hatch": None},
    "驗證": {"facecolor": "#d9d9d9", "hatch": None},
    "測試": {"facecolor": "white", "hatch": "////"},
}
PROTOCOLS = [  # name, purpose, (role, first month, last month)
    (
        "最終評估（final）",
        "模型比較、公平性量測、重現",
        [("訓練", 0, 5), ("測試", 6, 7)],
    ),
    (
        "開發（dev）",
        "調參、門檻與決策政策選擇",
        [("訓練", 0, 4), ("驗證", 5, 5), ("測試", 6, 7)],
    ),
]
HEIGHT = 0.62


def draw():
    use_serif()
    plt.rcParams["hatch.linewidth"] = 0.6
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    for row, (name, purpose, parts) in enumerate(PROTOCOLS):
        y = 1.45 - row * 1.15
        ax.text(-0.35, y + 0.42, name, ha="right", va="center", fontsize=11, color=INK)
        ax.text(-0.35, y + 0.14, purpose, ha="right", va="center", fontsize=9, color=INK)
        for role, first, last in parts:
            width = last - first + 1
            ax.add_patch(
                Rectangle((first, y), width, HEIGHT, edgecolor=INK, linewidth=0.9, **STYLE[role])
            )
            label = f"{role}（第 {first}–{last} 月）" if role == "訓練" else role
            ax.text(
                first + width / 2,
                y + HEIGHT / 2,
                label,
                ha="center",
                va="center",
                fontsize=10,
                color=INK,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}
                if role == "測試"
                else None,
            )
        for month in range(1, 8):  # month boundaries inside the bars
            ax.plot([month, month], [y, y + 0.08], color=INK, lw=0.6)
    for month in range(8):
        ax.text(month + 0.5, -0.02, f"{month}", ha="center", va="top", fontsize=9, color=INK)
    ax.text(4.0, -0.36, "月份（month）", ha="center", va="top", fontsize=9, color=INK)
    ax.set_xlim(-4.3, 8.1)
    ax.set_ylim(-0.65, 2.2)
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print(save_figure(draw(), "fig3_data_split", OUT))
