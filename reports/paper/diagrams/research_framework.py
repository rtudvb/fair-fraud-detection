"""Draw the research framework of chapter 3 (圖 3-1) in a formal-paper style.

Black lines on white, one concept per box, hypotheses written on the arrows and moderators
drawn as dashed arrows into the relationship they moderate. Details belong in the text.

Run: python reports/paper/diagrams/research_framework.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle

from frauddetect.viz import save_figure

PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "figures"
SERIF = PAPER / "fonts" / "NotoSerifCJKtc-Regular.otf"  # `make fonts`; 明體-like, as in theses
INK = "#000000"
GROUP = "#666666"


def use_serif() -> None:
    if SERIF.exists():
        font_manager.fontManager.addfont(str(SERIF))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(SERIF)).get_name()
    plt.rcParams["mathtext.fontset"] = "stix"  # Times-like math to match the serif text
    plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, size=11):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor=INK, linewidth=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK)


def group(ax, x, y, w, h, title):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor="none", edgecolor=GROUP, linewidth=0.9, ls=(0, (4, 3)))
    )
    ax.text(x + w / 2, y + h - 0.38, title, ha="center", va="center", fontsize=11, color=INK)


def arrow(ax, start, end, *, dashed=False, label=None, at=0.5, label_offset=(0.0, 0.0)):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            color=INK,
            linewidth=0.9,
            linestyle=(0, (4, 3)) if dashed else "-",
            shrinkA=0,
            shrinkB=0,
        )
    )
    if label:
        mx = start[0] + (end[0] - start[0]) * at  # label position along the arrow
        my = start[1] + (end[1] - start[1]) * at
        ax.text(
            mx + label_offset[0],
            my + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )


def draw():
    use_serif()
    fig, ax = plt.subplots(figsize=(8.2, 3.7))  # printed near full size: 9-10 pt text
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.8)
    ax.axis("off")

    w, h = 2.6, 1.0  # concept box
    mid = 4.6  # vertical centre line of the main flow

    # Data -> model generations.
    box(ax, 0.2, mid - 0.7, 2.6, 1.4, "BAF 開戶\n詐騙資料")
    group(ax, 3.8, 1.9, 3.6, 5.4, "模型世代")
    for y, name in zip(
        (5.4, 3.95, 2.5), ("梯度提升樹", "表格深度學習", "表格基礎模型"), strict=True
    ):
        box(ax, 4.3, y, w, h, name)
    arrow(ax, (2.8, mid), (3.8, mid))

    # Model generations -> model outcomes (H1, H2; cost is compared without a hypothesis).
    group(ax, 9.6, 1.9, 3.4, 5.4, "模型表現")
    outcomes = {"偵測效能": 5.4, "運算成本": 3.95, "年齡公平性": 2.5}
    for name, y in outcomes.items():
        box(ax, 10.0, y, w, h, name)
    start = (7.4, mid)
    arrow(ax, start, (10.0, 5.9), label="H1", at=0.25, label_offset=(-0.1, 0.3))
    arrow(ax, start, (10.0, 4.45))
    arrow(ax, start, (10.0, 3.0), label="H2", at=0.25, label_offset=(-0.1, -0.32))

    # Moderators: training-set size on H1, age-group definition on H2 (H3).
    box(ax, 7.35, 7.55, 2.7, 0.85, "訓練資料量", size=10.5)
    arrow(ax, (8.7, 7.55), (8.7, 5.32), dashed=True)
    box(ax, 7.35, 0.25, 2.7, 0.85, "年齡分組定義", size=10.5)
    arrow(ax, (8.7, 1.1), (8.7, 3.86), dashed=True, label="H3", at=0.5, label_offset=(0.38, 0.0))

    # Decision -> expected loss (H4) -> management implications.
    box(ax, 14.0, mid - 0.5, w, h, "三級分流決策")
    arrow(ax, (13.0, mid), (14.0, mid))
    box(ax, 14.0, 1.6, w, h, "期望損失")
    arrow(ax, (15.3, mid - 0.5), (15.3, 2.6), label="H4", label_offset=(0.38, 0.0))
    box(ax, 17.3, mid - 0.5, w, h, "管理意涵")
    arrow(ax, (16.6, mid), (17.3, mid))

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print(save_figure(draw(), "fig3_research_framework", OUT))
