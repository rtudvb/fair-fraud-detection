import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from frauddetect.models.registry import MODELS
from frauddetect.viz import FAMILY_COLORS, MODEL_MARKERS, save_figure, use_report_style


def test_every_model_has_family_color_and_marker():
    for name, spec in MODELS.items():
        assert spec.family in FAMILY_COLORS
        assert name in MODEL_MARKERS


def test_save_figure_writes_png_and_pdf(tmp_path):
    use_report_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    paths = save_figure(fig, "demo", tmp_path)
    plt.close(fig)
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
