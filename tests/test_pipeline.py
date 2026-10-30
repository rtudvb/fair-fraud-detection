"""End-to-end check: every experiment runs on synthetic data and writes its outputs."""

import pytest

from frauddetect.cli import main
from frauddetect.experiments import EXPERIMENTS, PIPELINE, get_runner, make_context


def test_pipeline_lists_every_experiment():
    assert set(PIPELINE) == set(EXPERIMENTS)
    for name in PIPELINE:
        assert callable(get_runner(name))


def test_unknown_experiment_raises():
    with pytest.raises(KeyError):
        get_runner("nope")


def test_quick_block_is_merged():
    quick = make_context("x", synthetic=True, quick=True)
    full = make_context("x", synthetic=True)
    assert quick.section("tuned")["n_trials"] < full.section("tuned")["n_trials"]
    assert quick.results_root.name == "_synthetic"


@pytest.mark.slow
def test_full_pipeline_on_synthetic_data(tmp_path, monkeypatch):
    import frauddetect.experiments.context as context

    original = context.make_context

    def isolated(*args, **kwargs):
        ctx = original(*args, **kwargs)
        ctx.config["paths"]["results"] = tmp_path
        return ctx

    monkeypatch.setattr("frauddetect.experiments.make_context", isolated)
    assert main(["run", "all", "--synthetic", "--quick", "--models", "logreg", "lightgbm"]) == 0
    root = tmp_path / "_synthetic"
    figures = {p.stem for p in (root / "figures").glob("*.png")}
    expected = {
        "e0_monthly",
        "e1_figure1",
        "e2_tpr",
        "e4_learning_curve",
        "e5_group_fpr",
        "e6_tradeoff",
        "e7_cost_breakdown",
        "e9_monthly",
        "e10_small_bank",
    }
    assert expected <= figures
    assert (root / "tables" / "e7_policies_test.md").exists()
    assert (root / "app" / "score_histogram.json").exists()
    assert (root / "monitor" / "meta.json").exists()
