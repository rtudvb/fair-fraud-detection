"""Random hyperparameter sampling from the YAML search-space format.

Space format (see ``configs/models/lightgbm_baf_space.yaml``)::

    name: {type: int | float, low: ..., high: ..., log: bool}
    name: {type: categorical, choices: [...]}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def sample_params(space: Mapping[str, Mapping[str, Any]], rng: np.random.Generator) -> dict:
    """Draw one configuration; log-scaled ranges are sampled uniformly in log space."""
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "categorical":
            choices = list(spec["choices"])
            params[name] = choices[int(rng.integers(len(choices)))]
            continue
        low, high = float(spec["low"]), float(spec["high"])
        if low > high:
            raise ValueError(f"{name}: low must not exceed high")
        if spec.get("log", False):
            value = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        else:
            value = float(rng.uniform(low, high))
        if kind == "int":
            params[name] = int(np.clip(round(value), low, high))
        elif kind == "float":
            params[name] = value
        else:
            raise ValueError(f"{name}: unknown type {kind!r}")
    return params


def to_lightgbm_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Translate BAF-paper names to LightGBM >= 4 arguments.

    GOSS moved from ``boosting_type="goss"`` to ``data_sample_strategy="goss"``.
    """
    out = dict(params)
    if out.get("boosting_type") == "goss":
        out["boosting_type"] = "gbdt"
        out["data_sample_strategy"] = "goss"
    return out


def suggest_params(trial: Any, space: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Optuna counterpart of :func:`sample_params` for the same YAML space format.

    Categorical choices that are lists (e.g. MLP layer sizes) are offered to Optuna as
    strings and decoded back, because Optuna only stores scalar choices.
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        if kind == "categorical":
            choices = list(spec["choices"])
            if any(isinstance(c, list | tuple) for c in choices):
                labels = {"-".join(map(str, c)): list(c) for c in choices}
                params[name] = labels[trial.suggest_categorical(name, list(labels))]
            else:
                params[name] = trial.suggest_categorical(name, choices)
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), log=bool(spec.get("log", False))
            )
        elif kind == "float":
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
            )
        else:
            raise ValueError(f"{name}: unknown type {kind!r}")
    return params
