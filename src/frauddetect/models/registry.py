"""One registry for all ten benchmark models, behind the scikit-learn interface.

``build_model(name)`` returns an estimator whose ``fit(X, y)`` takes the shared pandas frame
(categoricals in ``category`` dtype) and whose ``predict_proba(X)[:, 1]`` is the fraud
score, so every experiment runs the same evaluation code. Heavy libraries are imported
only when their model is built; install them with the matching extra (see ``pyproject``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from frauddetect.models.preprocessing import CategoryAligner, one_hot_and_scale, ordinal_encode

Builder = Callable[[dict[str, Any], int], BaseEstimator]


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry: family for grouping in figures, extra needed to install it."""

    name: str
    family: str  # "classical" | "deep_learning" | "foundation"
    extra: str
    builder: Builder


def _logreg(params: dict, seed: int) -> BaseEstimator:
    from sklearn.linear_model import LogisticRegression

    return Pipeline(
        [
            ("align", CategoryAligner()),
            ("prep", one_hot_and_scale()),
            ("clf", LogisticRegression(random_state=seed, **params)),
        ]
    )


def _random_forest(params: dict, seed: int) -> BaseEstimator:
    from sklearn.ensemble import RandomForestClassifier

    return Pipeline(
        [
            ("align", CategoryAligner()),
            ("prep", ordinal_encode()),
            ("clf", RandomForestClassifier(random_state=seed, **params)),
        ]
    )


def _xgboost(params: dict, seed: int) -> BaseEstimator:
    from xgboost import XGBClassifier

    return Pipeline(
        [
            ("align", CategoryAligner()),
            ("clf", XGBClassifier(enable_categorical=True, random_state=seed, **params)),
        ]
    )


def _lightgbm(params: dict, seed: int) -> BaseEstimator:
    from lightgbm import LGBMClassifier

    return Pipeline(
        [("align", CategoryAligner()), ("clf", LGBMClassifier(random_state=seed, **params))]
    )


class CatBoostWithCategories(BaseEstimator, ClassifierMixin):
    """CatBoost needs the categorical column names up front; read them from the dtypes."""

    def __init__(self, random_state: int = 0, **params: Any) -> None:
        self.random_state = random_state
        self.params = params

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {"random_state": self.random_state, **self.params}

    def set_params(self, **params: Any) -> CatBoostWithCategories:
        self.random_state = params.pop("random_state", self.random_state)
        self.params.update(params)
        return self

    def _to_catboost(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cat_features_:
            values = X[col].astype(object)
            X[col] = values.where(values.notna(), "__missing__")  # CatBoost rejects NaN
        return X

    def fit(self, X: pd.DataFrame, y) -> CatBoostWithCategories:
        from catboost import CatBoostClassifier

        self.cat_features_ = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        params = {"allow_writing_files": False, **self.params}  # no catboost_info/ litter
        self.model_ = CatBoostClassifier(
            random_seed=self.random_state, cat_features=self.cat_features_, **params
        )
        self.model_.fit(self._to_catboost(X), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self._to_catboost(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def _catboost(params: dict, seed: int) -> BaseEstimator:
    return Pipeline(
        [("align", CategoryAligner()), ("clf", CatBoostWithCategories(random_state=seed, **params))]
    )


def _mlp(params: dict, seed: int) -> BaseEstimator:
    from sklearn.neural_network import MLPClassifier

    return Pipeline(
        [
            ("align", CategoryAligner()),
            ("prep", one_hot_and_scale()),
            ("clf", MLPClassifier(random_state=seed, **params)),
        ]
    )


def _tabm(params: dict, seed: int) -> BaseEstimator:
    from pytabkit import TabM_D_Classifier

    return Pipeline(
        [("align", CategoryAligner()), ("clf", TabM_D_Classifier(random_state=seed, **params))]
    )


def _realmlp(params: dict, seed: int) -> BaseEstimator:
    from pytabkit import RealMLP_TD_Classifier

    return Pipeline(
        [("align", CategoryAligner()), ("clf", RealMLP_TD_Classifier(random_state=seed, **params))]
    )


def _tabpfn(params: dict, seed: int) -> BaseEstimator:
    from tabpfn import TabPFNClassifier

    return Pipeline(
        [("align", CategoryAligner()), ("clf", TabPFNClassifier(random_state=seed, **params))]
    )


def _tabicl(params: dict, seed: int) -> BaseEstimator:
    from tabicl import TabICLClassifier

    return Pipeline(
        [("align", CategoryAligner()), ("clf", TabICLClassifier(random_state=seed, **params))]
    )


MODELS: dict[str, ModelSpec] = {
    spec.name: spec
    for spec in (
        ModelSpec("logreg", "classical", "core", _logreg),
        ModelSpec("random_forest", "classical", "core", _random_forest),
        ModelSpec("xgboost", "classical", "gbdt", _xgboost),
        ModelSpec("lightgbm", "classical", "gbdt", _lightgbm),
        ModelSpec("catboost", "classical", "gbdt", _catboost),
        ModelSpec("mlp", "deep_learning", "core", _mlp),
        ModelSpec("tabm", "deep_learning", "dl", _tabm),
        ModelSpec("realmlp", "deep_learning", "dl", _realmlp),
        ModelSpec("tabpfn", "foundation", "fm", _tabpfn),
        ModelSpec("tabicl", "foundation", "fm", _tabicl),
    )
}

FAMILIES: tuple[str, ...] = ("classical", "deep_learning", "foundation")


def build_model(
    name: str, params: Mapping[str, Any] | None = None, *, random_state: int = 0
) -> BaseEstimator:
    """Instantiate a registered model. Raises ``ImportError`` naming the extra to install."""
    if name not in MODELS:
        raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODELS)}")
    spec = MODELS[name]
    try:
        return spec.builder(dict(params or {}), random_state)
    except ImportError as exc:
        raise ImportError(
            f"Model {name!r} needs optional dependencies: pip install -e '.[{spec.extra}]'"
        ) from exc


def predict_scores(
    model: BaseEstimator, X: pd.DataFrame, batch_size: int | None = None
) -> np.ndarray:
    """Fraud score in [0, 1]: the predicted probability of the positive class.

    ``batch_size`` scores large frames in chunks to bound memory (foundation models).
    """
    if batch_size is None or len(X) <= batch_size:
        return np.asarray(model.predict_proba(X))[:, 1]
    chunks = [
        np.asarray(model.predict_proba(X.iloc[start : start + batch_size]))[:, 1]
        for start in range(0, len(X), batch_size)
    ]
    return np.concatenate(chunks)
