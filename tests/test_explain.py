import numpy as np
import pandas as pd
import pytest

from frauddetect.data.split import temporal_split
from frauddetect.explain.rules import fit_surrogate, leaf_rules, one_hot_frame


def test_one_hot_frame_aligns_columns():
    train = pd.DataFrame({"c": pd.Categorical(["a", "b"]), "x": [1.0, 2.0]})
    test = pd.DataFrame({"c": pd.Categorical(["a"]), "x": [3.0]})
    a = one_hot_frame(train)
    b = one_hot_frame(test, list(a.columns))
    assert list(a.columns) == list(b.columns) == ["x", "c=a", "c=b"]
    assert b.loc[0, "c=b"] == 0


def test_surrogate_recovers_a_simple_rule(baf_frame):
    split = temporal_split(baf_frame, train=range(6), test=[6, 7])
    X_train, _ = split.xy("train")
    X_test, y_test = split.xy("test")
    flags_train = (X_train["velocity_6h"] > 1.0).to_numpy()
    flags_test = (X_test["velocity_6h"] > 1.0).to_numpy()
    surrogate = fit_surrogate(X_train, flags_train, X_test, flags_test, max_depth=2)
    assert surrogate.fidelity > 0.95
    assert "velocity_6h" in surrogate.text()
    rules = leaf_rules(surrogate, X_test, y_test, flags_test)
    assert rules["coverage"].sum() == pytest.approx(1.0)
    assert set(rules.columns) >= {"rule", "fraud_rate", "model_flag_rate"}


def test_shap_values_on_lightgbm(baf_frame):
    pytest.importorskip("shap")
    pytest.importorskip("lightgbm")
    from frauddetect.experiments.explain_exp import importance_table, shap_values
    from frauddetect.models.registry import build_model

    split = temporal_split(baf_frame, train=range(6), test=[6, 7])
    X_train, y_train = split.xy("train")
    X_test, _ = split.xy("test")
    model = build_model("lightgbm", {"n_estimators": 30, "verbose": -1}).fit(X_train, y_train)
    values = shap_values(model, X_test.iloc[:200])
    assert values.shape == (200, X_test.shape[1])
    table = importance_table(values, X_test.iloc[:200])
    assert table.iloc[0]["feature"] in {"velocity_6h", "name_email_similarity"}
    assert np.isfinite(table["mean_abs_shap"]).all()
