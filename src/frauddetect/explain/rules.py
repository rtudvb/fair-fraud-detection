"""Turn a fraud model into readable risk-control rules with a shallow surrogate tree.

The surrogate learns to imitate the model's *flags* (not the labels), so its rules describe
what the model does. Fidelity — how often the surrogate agrees with the model — is reported
with the rules; low fidelity means the rules must not be presented as "the model's logic".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeClassifier, _tree, export_text


def one_hot_frame(X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Numeric columns as-is, categorical columns as ``name=level`` indicators.

    Pass the training ``columns`` when encoding test data so both share one layout.
    """
    cat_cols = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
    encoded = pd.get_dummies(X, columns=cat_cols, prefix_sep="=", dtype=float)
    if columns is not None:
        encoded = encoded.reindex(columns=columns, fill_value=0.0)
    return encoded


@dataclass
class SurrogateRules:
    tree: DecisionTreeClassifier
    columns: list[str]
    fidelity: float  # accuracy of the surrogate against the model's flags
    balanced_fidelity: float  # same, averaged over flagged / not-flagged (flags are rare)
    flag_recall: float  # share of model flags the surrogate also flags
    flag_precision: float  # share of surrogate flags the model also flags

    def text(self) -> str:
        return export_text(self.tree, feature_names=self.columns, decimals=2)


def fit_surrogate(
    X_train: pd.DataFrame,
    flags_train: np.ndarray,
    X_test: pd.DataFrame,
    flags_test: np.ndarray,
    *,
    max_depth: int = 3,
    min_leaf_share: float = 0.002,
    seed: int = 0,
) -> SurrogateRules:
    """Fit a depth-limited tree imitating the model's flags; fidelity is measured on test."""
    A = one_hot_frame(X_train)
    B = one_hot_frame(X_test, list(A.columns))
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(20, int(min_leaf_share * len(A))),
        class_weight="balanced",
        random_state=seed,
    ).fit(A, flags_train)
    pred = tree.predict(B).astype(bool)
    flags_test = np.asarray(flags_test).astype(bool)
    return SurrogateRules(
        tree=tree,
        columns=list(A.columns),
        fidelity=float(np.mean(pred == flags_test)),
        balanced_fidelity=float(balanced_accuracy_score(flags_test, pred)),
        flag_recall=float(pred[flags_test].mean()) if flags_test.any() else float("nan"),
        flag_precision=float(flags_test[pred].mean()) if pred.any() else float("nan"),
    )


def leaf_rules(
    surrogate: SurrogateRules, X: pd.DataFrame, y_true: np.ndarray, flags: np.ndarray
) -> pd.DataFrame:
    """One row per leaf: readable condition, coverage, model-flag rate and actual fraud rate."""
    tree = surrogate.tree.tree_
    names = surrogate.columns
    paths: dict[int, list[str]] = {}

    def walk(node: int, conditions: list[str]) -> None:
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            paths[node] = conditions
            return
        name, threshold = names[tree.feature[node]], tree.threshold[node]
        if "=" in name:  # one-hot indicator: <= 0.5 means "not this level"
            left, right = f"{name.replace('=', ' ≠ ')}", f"{name.replace('=', ' = ')}"
        else:
            left, right = f"{name} ≤ {threshold:.2f}", f"{name} > {threshold:.2f}"
        walk(tree.children_left[node], [*conditions, left])
        walk(tree.children_right[node], [*conditions, right])

    walk(0, [])
    encoded = one_hot_frame(X, names)
    leaves = surrogate.tree.apply(encoded)
    y = np.asarray(y_true).astype(bool)
    flags = np.asarray(flags).astype(bool)
    rows = []
    for leaf, conditions in paths.items():
        mask = leaves == leaf
        if not mask.any():
            continue
        rows.append(
            {
                "rule": " 且 ".join(conditions) or "（全部）",
                "surrogate_flags": bool(surrogate.tree.tree_.value[leaf][0].argmax() == 1),
                "coverage": mask.mean(),
                "model_flag_rate": flags[mask].mean(),
                "fraud_rate": y[mask].mean(),
                "fraud_share": y[mask].sum() / max(y.sum(), 1),
            }
        )
    return pd.DataFrame(rows).sort_values("fraud_rate", ascending=False, ignore_index=True)
