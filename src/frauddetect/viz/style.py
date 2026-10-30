"""One figure style for the whole report (matplotlib, light print surface).

Colors follow a validated categorical palette (colorblind-safe for up to three series on
every pair) and a one-hue ordinal ramp for the decision tiers. Identity is never carried by
color alone: models within a family differ by marker, and figures keep a legend or direct
labels. Chinese labels need a CJK font; the first installed one from ``CJK_FONTS`` is used.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

log = logging.getLogger(__name__)

# Ink and chrome
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Categorical slots 1-3 (validated all-pairs: CVD dE >= 9.2, normal-vision dE >= 24).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

FAMILY_COLORS = {"classical": BLUE, "deep_learning": ORANGE, "foundation": AQUA}
FAMILY_LABELS = {"classical": "傳統機器學習", "deep_learning": "深度學習", "foundation": "基礎模型"}
GROUP_COLORS = {"younger": BLUE, "older": ORANGE}
GROUP_LABELS = {"younger": "未滿 50 歲", "older": "50 歲以上"}

# Ordinal ramp for tiers 1 -> 3 (one hue, light to dark; light end clears 2:1 on white).
TIER_COLORS = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}
TIER_LABELS = {1: "一級：正常開戶", 2: "二級：限額＋加強驗證", 3: "三級：人工審核"}

# Models inside one family share the family color and differ by marker.
MODEL_MARKERS = {
    "logreg": "o",
    "random_forest": "s",
    "xgboost": "^",
    "lightgbm": "D",
    "catboost": "v",
    "mlp": "o",
    "tabm": "s",
    "realmlp": "^",
    "tabpfn": "o",
    "tabicl": "s",
}

CJK_FONTS = (
    "Noto Sans CJK TC",
    "Noto Sans TC",
    "Source Han Sans TC",
    "Microsoft JhengHei",
    "PingFang TC",
    "Heiti TC",
    "Arial Unicode MS",
)


def register_repo_fonts(font_dir: Path | None = None) -> None:
    """Make fonts downloaded by ``make fonts`` (``reports/paper/fonts``) visible to matplotlib."""
    if font_dir is None:
        from frauddetect.config import project_root

        try:
            font_dir = project_root() / "reports" / "paper" / "fonts"
        except FileNotFoundError:  # installed outside the repository
            return
    for path in sorted(Path(font_dir).glob("*.[ot]tf")):
        font_manager.fontManager.addfont(str(path))


def available_cjk_font() -> str | None:
    """Name of the first installed CJK font, or ``None``."""
    register_repo_fonts()
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in CJK_FONTS if name in installed), None)


def use_report_style() -> None:
    """Apply the report style globally (call once at the top of a notebook or script)."""
    cjk = available_cjk_font()
    if cjk is None:
        log.warning("No CJK font found; Chinese labels will not render. Install Noto Sans CJK TC.")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [*([cjk] if cjk else []), "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.labelcolor": INK_SECONDARY,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, AQUA]),
        }
    )


def save_figure(fig: plt.Figure, name: str, directory: str | Path) -> list[Path]:
    """Save ``fig`` as PNG (300 dpi) and vector PDF under ``directory``; return the paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{name}.png", directory / f"{name}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=300)  # explicit: callers need not load the report style
    return paths
