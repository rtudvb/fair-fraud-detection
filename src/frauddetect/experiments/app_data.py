"""E11 — export the aggregated score histogram that powers the Streamlit demo.

Only bin counts per (label, age group) are written to ``app/data/``; no application rows.
"""

from __future__ import annotations

import logging

from frauddetect.evaluation.fairness import age_groups
from frauddetect.experiments.context import ExperimentContext
from frauddetect.experiments.scoring import get_scores
from frauddetect.simulator import ScoreHistogram

log = logging.getLogger(__name__)


def run(ctx: ExperimentContext) -> ScoreHistogram:
    section = ctx.section("app")
    model = section["model"]
    test = get_scores(ctx, model, protocol="dev", variant=section.get("variant", "base")).part(
        "test"
    )
    groups = age_groups(test["age"], int(ctx.config["fairness"]["age_cutoff"]))
    histogram = ScoreHistogram.from_scores(
        test["y"],
        test["score"],
        groups,
        n_bins=int(section.get("n_bins", 1000)),
        model=model,
        period="BAF Base 第 6–7 月",
    )
    folder = ctx.results_root / "app" if ctx.synthetic else ctx.root / "app" / "data"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "score_histogram.json"
    histogram.to_json(path)
    log.info("demo data written to %s (%d bins)", path, histogram.n_bins)
    return histogram
