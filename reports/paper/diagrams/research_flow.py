"""Draw the research flow of chapter 1 (圖 1-1) in the formal style of 圖 3-1.

A vertical flowchart: one short step per box, the experiment step fans out into its three
analyses and merges again, and brackets on the right map the steps to chapters.

Run: python reports/paper/diagrams/research_flow.py
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from research_framework import INK, arrow, box, use_serif

from frauddetect.viz import save_figure

OUT = Path(__file__).resolve().parents[1] / "figures"

CENTER, WIDTH, HEIGHT = 4.6, 4.6, 1.0
STEPS = {  # name -> top y of the box
    "研究背景與動機": 19.4,
    "研究目的與問題": 17.5,
    "文獻探討": 15.6,
    "研究架構與假說": 13.7,
    "資料前處理與實驗設計": 11.8,
    "假說檢驗與知識發現": 7.0,
    "管理意涵與建議": 5.1,
    "結論與後續研究": 3.2,
}
BRANCHES = ("模型效能\n與成本", "年齡\n公平性", "三級分流\n決策")
BRANCH_TOP, BRANCH_HEIGHT, BRANCH_WIDTH = 9.5, 1.3, 2.6
CHAPTERS = [  # label, first step, last step (or the branch row)
    ("第一章", "研究背景與動機", "研究目的與問題"),
    ("第二章", "文獻探討", "文獻探討"),
    ("第三章", "研究架構與假說", "資料前處理與實驗設計"),
    ("第四章", "branches", "假說檢驗與知識發現"),
    ("第五章", "管理意涵與建議", "管理意涵與建議"),
    ("第六章", "結論與後續研究", "結論與後續研究"),
]


def span(step: str) -> tuple[float, float]:
    """(top, bottom) of a step or of the branch row."""
    if step == "branches":
        return BRANCH_TOP, BRANCH_TOP - BRANCH_HEIGHT
    return STEPS[step], STEPS[step] - HEIGHT


def draw():
    use_serif()
    fig, ax = plt.subplots(figsize=(5.6, 8.2))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(1.9, 19.8)
    ax.axis("off")

    names = list(STEPS)
    for name, top in STEPS.items():
        box(ax, CENTER - WIDTH / 2, top - HEIGHT, WIDTH, HEIGHT, name)
    # Straight arrows between consecutive steps, except around the branch row.
    for upper, lower in pairwise(names):
        if upper == "資料前處理與實驗設計":
            continue
        arrow(ax, (CENTER, STEPS[upper] - HEIGHT), (CENTER, STEPS[lower]))

    # Fan out into the three analyses and merge again.
    xs = [CENTER - 3.0, CENTER, CENTER + 3.0]
    for x, name in zip(xs, BRANCHES, strict=True):
        box(ax, x - BRANCH_WIDTH / 2, BRANCH_TOP - BRANCH_HEIGHT, BRANCH_WIDTH, BRANCH_HEIGHT, name)
    split_y = BRANCH_TOP + 0.55
    ax.plot([CENTER, CENTER], [STEPS["資料前處理與實驗設計"] - HEIGHT, split_y], color=INK, lw=0.9)
    ax.plot([xs[0], xs[-1]], [split_y, split_y], color=INK, lw=0.9)
    for x in xs:
        arrow(ax, (x, split_y), (x, BRANCH_TOP))
    merge_y = BRANCH_TOP - BRANCH_HEIGHT - 0.55
    for x in xs:
        ax.plot([x, x], [BRANCH_TOP - BRANCH_HEIGHT, merge_y], color=INK, lw=0.9)
    ax.plot([xs[0], xs[-1]], [merge_y, merge_y], color=INK, lw=0.9)
    arrow(ax, (CENTER, merge_y), (CENTER, STEPS["假說檢驗與知識發現"]))

    # Chapter brackets on the right.
    bracket_x = 9.7
    for label, first, last in CHAPTERS:
        top, bottom = span(first)[0], span(last)[1]
        ax.plot(
            [bracket_x - 0.2, bracket_x, bracket_x, bracket_x - 0.2],
            [top, top, bottom, bottom],
            color=INK,
            lw=0.8,
        )
        ax.text(bracket_x + 0.3, (top + bottom) / 2, label, va="center", fontsize=11, color=INK)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print(save_figure(draw(), "fig1_research_flow", OUT))
