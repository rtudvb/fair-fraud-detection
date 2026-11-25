from pathlib import Path

import numpy as np
import pytest

from frauddetect.decision.three_tier import TierCosts, evaluate_policy
from frauddetect.simulator import ScoreHistogram

COSTS = TierCosts(200_000, 50_000, 0.5, 0.9, 50, 300, 100, 2000)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 30000
    groups = np.where(rng.random(n) < 0.25, "older", "younger")
    y = (rng.random(n) < 0.02).astype(int)
    scores = rng.random(n) * 0.5 + 0.5 * y * rng.random(n)
    return y, scores, groups


def test_rates_match_raw_scores_at_bin_edges(data):
    y, s, g = data
    hist = ScoreHistogram.from_scores(y, s, g, n_bins=200)
    b = hist.bin_for_alert_rate(0.05)
    threshold = hist.edges[b]
    flagged = s >= threshold
    rates = hist.rates(b)
    assert rates["alert_rate"] == pytest.approx(flagged.mean(), abs=1e-9)
    assert rates["tpr"] == pytest.approx(flagged[y == 1].mean(), abs=1e-9)
    assert rates["fpr_older"] == pytest.approx(flagged[(y == 0) & (g == "older")].mean())
    assert rates["alert_rate"] <= 0.05


def test_three_tier_cost_matches_decision_module(data):
    y, s, g = data
    hist = ScoreHistogram.from_scores(y, s, g, n_bins=300)
    b1, b2 = hist.bin_for_alert_rate(0.1), hist.bin_for_alert_rate(0.02)
    result = hist.three_tier(b1, b2, COSTS)
    exact = evaluate_policy(y, s, hist.edges[b1], hist.edges[b2], COSTS)
    assert result["cost_per_app"] == pytest.approx(exact["cost_per_application"])
    assert result["tier3_share"] == pytest.approx(exact["review_rate"])
    caught = result["fraud_tier2"] * COSTS.verify_catch + result["fraud_tier3"] * COSTS.review_catch
    assert result["fraud_caught"] == pytest.approx(caught)
    assert result["fraud_caught"] <= result["fraud_tier2"] + result["fraud_tier3"]
    with pytest.raises(ValueError):
        hist.three_tier(b2, b1, COSTS)


def test_json_round_trip(tmp_path: Path, data):
    hist = ScoreHistogram.from_scores(*data, n_bins=50, model="m", period="p")
    hist.to_json(tmp_path / "h.json")
    back = ScoreHistogram.from_json(tmp_path / "h.json")
    assert back.model == "m" and back.n_bins == hist.n_bins
    assert back.total() == len(data[0])


def test_streamlit_app_renders_demo_data():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).parents[1] / "app" / "streamlit_app.py"))
    app.run(timeout=60)
    assert not app.exception
    assert app.title[0].value == "開戶風控決策支援系統"
    page = " ".join(block.value for block in app.markdown)
    for label in (
        "詐騙預期攔截率",
        "正常客戶受影響比例",
        "年齡公平性",
        "每件申請期望成本",
        "人力上限",
    ):
        assert label in page, label
