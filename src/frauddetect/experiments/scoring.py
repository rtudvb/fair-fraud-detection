"""Fit a model on a temporal split and keep its scores, cached on disk.

Two protocols (proposal §6.2):

* ``final`` — train on months 0-5, score the test months 6-7 (model comparison, fairness).
* ``dev``   — train on months 0-4, score validation month 5 and test months 6-7
  (thresholds and policies are chosen on month 5 and evaluated on 6-7).

Scores are saved as Parquet with the label, age and month of every scored row, so later
experiments (fairness, decision, monitoring, demo) reuse them without retraining.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from frauddetect.config import load_yaml
from frauddetect.data.schema import LABEL, MONTH, PROTECTED
from frauddetect.data.split import stratified_subsample
from frauddetect.evaluation.fairness import age_groups, fpr_ratio, tpr_ratio
from frauddetect.evaluation.metrics import summary_metrics, threshold_at_fpr
from frauddetect.experiments.context import TORCH_MODELS, ExperimentContext
from frauddetect.models.registry import build_model, predict_scores

log = logging.getLogger(__name__)

PROTOCOLS: dict[str, dict[str, Any]] = {
    "final": {"train": "reproduction", "parts": {"test": "test"}},
    "dev": {"train": "main", "parts": {"valid": "valid", "test": "test"}},
}


@dataclass
class ScoreRun:
    """Scores of one fitted model; ``scores`` has columns part, y, score, age, month."""

    model: str
    protocol: str
    seed: int
    params: dict[str, Any]
    scores: pd.DataFrame
    n_train: int
    fit_seconds: float
    predict_seconds: float

    def part(self, name: str) -> pd.DataFrame:
        return self.scores.loc[self.scores["part"] == name].reset_index(drop=True)

    @property
    def meta(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "protocol": self.protocol,
            "seed": self.seed,
            "params": self.params,
            "n_train": self.n_train,
            "fit_seconds": round(self.fit_seconds, 2),
            "predict_seconds": round(self.predict_seconds, 2),
        }


def default_params(ctx: ExperimentContext, model: str) -> dict[str, Any]:
    """Default-track parameters from ``configs/models/default.yaml``."""
    config = load_yaml(ctx.root / "configs" / "models" / "default.yaml")
    return dict(config["models"].get(model, {}).get("params", {}))


def runtime_params(ctx: ExperimentContext, model: str, params: dict[str, Any]) -> dict[str, Any]:
    """Add the run's device to PyTorch-based models unless the caller set one."""
    params = dict(params)
    if model in TORCH_MODELS:
        params.setdefault("device", ctx.device)
    return params


def fit_and_score(
    ctx: ExperimentContext,
    model: str,
    train: pd.DataFrame,
    parts: dict[str, pd.DataFrame],
    *,
    params: dict[str, Any] | None = None,
    seed: int = 0,
    max_train_rows: int | None = None,
) -> ScoreRun:
    """Train ``model`` on ``train`` (optionally subsampled) and score every frame in ``parts``."""
    params = runtime_params(ctx, model, default_params(ctx, model) if params is None else params)
    if max_train_rows is not None:
        train = stratified_subsample(train, max_train_rows, seed=seed)
    X_train, y_train = train.drop(columns=[LABEL, MONTH]), train[LABEL].to_numpy()

    log.info("fit %-14s n=%-8d params=%s", model, len(train), params)
    start = time.perf_counter()
    estimator = build_model(model, params, random_state=seed).fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    batch = ctx.config["experiments"].get("runtime", {}).get("predict_batch_size")
    frames, start = [], time.perf_counter()
    for name, frame in parts.items():
        frames.append(
            pd.DataFrame(
                {
                    "part": name,
                    "y": frame[LABEL].to_numpy(),
                    "score": predict_scores(
                        estimator, frame.drop(columns=[LABEL, MONTH]), batch_size=batch
                    ),
                    "age": frame[PROTECTED].to_numpy(),
                    "month": frame[MONTH].to_numpy(),
                }
            )
        )
    return ScoreRun(
        model=model,
        protocol="custom",
        seed=seed,
        params=params,
        scores=pd.concat(frames, ignore_index=True),
        n_train=len(train),
        fit_seconds=fit_seconds,
        predict_seconds=time.perf_counter() - start,
    )


def get_scores(
    ctx: ExperimentContext,
    model: str,
    *,
    protocol: str = "final",
    variant: str = "base",
    seed: int = 0,
    params: dict[str, Any] | None = None,
    tag: str = "default",
    refresh: bool = False,
) -> ScoreRun:
    """Cached :func:`fit_and_score` on a named protocol (``final`` or ``dev``).

    ``tag`` separates parameter sets (``default`` vs ``tuned``) in the cache.
    """
    if protocol not in PROTOCOLS:
        raise KeyError(f"unknown protocol {protocol!r}")
    folder = ctx.scores_dir / variant / protocol / tag
    folder.mkdir(parents=True, exist_ok=True)
    stem = folder / f"{model}_s{seed}"
    parquet, meta_path = stem.with_suffix(".parquet"), stem.with_suffix(".json")
    if parquet.exists() and meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return ScoreRun(
            model=model,
            protocol=protocol,
            seed=seed,
            params=meta["params"],
            scores=pd.read_parquet(parquet),
            n_train=meta["n_train"],
            fit_seconds=meta["fit_seconds"],
            predict_seconds=meta["predict_seconds"],
        )

    from frauddetect.data.split import split_from_config

    spec = PROTOCOLS[protocol]
    split = split_from_config(ctx.load(variant), ctx.config, spec["train"])
    parts = {name: getattr(split, attr) for name, attr in spec["parts"].items()}
    run = fit_and_score(
        ctx,
        model,
        split.train,
        parts,
        params=params,
        seed=seed,
        max_train_rows=ctx.max_train_rows(model),
    )
    run.protocol = protocol
    run.scores.to_parquet(parquet, index=False)
    meta_path.write_text(json.dumps(run.meta, indent=2, default=str), encoding="utf-8")
    return run


def evaluate_scores(
    y: np.ndarray, scores: np.ndarray, age: np.ndarray, *, target_fpr: float, cutoff: int
) -> dict[str, float]:
    """Performance at the BAF operating point plus age fairness of the resulting flags."""
    metrics = summary_metrics(y, scores, target_fpr)
    flagged = scores >= threshold_at_fpr(y, scores, target_fpr)
    groups = age_groups(age, cutoff)
    metrics["fpr_ratio"] = fpr_ratio(y, flagged, groups)
    metrics["tpr_ratio"] = tpr_ratio(y, flagged, groups)
    return metrics
