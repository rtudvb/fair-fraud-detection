"""Draw the research schedule of chapter 1 (圖 1-2) as a Gantt chart, in the formal style.

Run: python reports/paper/diagrams/gantt.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from research_framework import INK, use_serif

from frauddetect.viz import save_figure

OUT = Path(__file__).resolve().parents[1] / "figures"
YEAR = 2026
TASKS = [  # name, first day, last day (month, day)
    ("文獻回顧與研究計畫", (10, 2), (10, 14)),
    ("環境建置與實驗程式開發", (10, 8), (10, 26)),
    ("資料下載與探索分析", (10, 20), (10, 25)),
    ("實驗執行（E0–E10）", (10, 24), (11, 4)),
    ("結果分析與三級分流決策", (10, 28), (11, 10)),
    ("決策支援系統開發", (10, 28), (11, 22)),
    ("報告撰寫", (10, 25), (11, 20)),
    ("校對與定稿", (11, 16), (11, 28)),
]


def draw():
    use_serif()
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for row, (_name, first, last) in enumerate(TASKS):
        start = date(YEAR, *first)
        days = (date(YEAR, *last) - start).days + 1
        ax.barh(row, days, left=start, height=0.5, color="#d9d9d9", edgecolor=INK, lw=0.8)
    ax.set_yticks(range(len(TASKS)), [name for name, _, _ in TASKS], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(date(YEAR, 10, 1), date(YEAR, 12, 1))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.FR))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color="#bbbbbb", lw=0.5, ls=(0, (3, 3)))
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(f"{YEAR} 年（每週五）", fontsize=10)
    end = date(YEAR, 11, 28) + timedelta(days=1)
    ax.axvline(end, color=INK, lw=0.8, ls="--")
    ax.text(end, -0.9, "定稿 11/28", ha="center", fontsize=9, color=INK)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    print(save_figure(draw(), "fig1_gantt", OUT))
