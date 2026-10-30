"""Experiment runners E0-E11 (proposal §7). Run with ``python -m frauddetect run <name>``.

Every runner takes an :class:`~frauddetect.experiments.context.ExperimentContext`, writes
tables and figures for the report, and records metadata (hardware, versions, config).
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module

from frauddetect.experiments.context import ExperimentContext, make_context

# name -> (module, function). Imported lazily so `run eda` does not import torch.
EXPERIMENTS: dict[str, tuple[str, str]] = {
    "eda": ("frauddetect.experiments.eda", "run"),
    "reproduce": ("frauddetect.experiments.reproduce", "run"),
    "benchmark": ("frauddetect.experiments.benchmark", "run_default"),
    "tuned": ("frauddetect.experiments.benchmark", "run_tuned"),
    "learning-curve": ("frauddetect.experiments.benchmark", "run_learning_curve"),
    "fairness": ("frauddetect.experiments.fairness_exp", "run"),
    "decision": ("frauddetect.experiments.decision_exp", "run"),
    "explain": ("frauddetect.experiments.explain_exp", "run"),
    "monitor": ("frauddetect.experiments.monitoring_exp", "run"),
    "small-bank": ("frauddetect.experiments.small_bank", "run"),
    "app-data": ("frauddetect.experiments.app_data", "run"),
}

# Order of `run all`: later steps reuse cached scores of earlier ones.
PIPELINE: tuple[str, ...] = (
    "eda",
    "reproduce",
    "benchmark",
    "tuned",
    "learning-curve",
    "fairness",
    "decision",
    "explain",
    "monitor",
    "small-bank",
    "app-data",
)


def get_runner(name: str) -> Callable[[ExperimentContext], object]:
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}")
    module, func = EXPERIMENTS[name]
    return getattr(import_module(module), func)


__all__ = ["EXPERIMENTS", "PIPELINE", "ExperimentContext", "get_runner", "make_context"]
