from collections import Counter

import numpy as np
import pandas as pd
import pytest

from frauddetect.cli import main
from frauddetect.config import load_yaml, project_root
from frauddetect.data.split import temporal_split
from frauddetect.models.preprocessing import CategoryAligner
from frauddetect.models.registry import MODELS, build_model, predict_scores
from frauddetect.models.search import sample_params, to_lightgbm_params


def test_registry_has_ten_models_in_three_families():
    assert len(MODELS) == 10
    families = Counter(spec.family for spec in MODELS.values())
    assert families == {"classical": 5, "deep_learning": 3, "foundation": 2}


def test_default_config_covers_registry():
    config = load_yaml(project_root() / "configs" / "models" / "default.yaml")
    assert set(config["models"]) == set(MODELS)
    for name, entry in config["models"].items():
        assert entry["family"] == MODELS[name].family


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("gpt")


def test_category_aligner_maps_unseen_levels_to_missing():
    train = pd.DataFrame({"c": pd.Categorical(["a", "b"]), "x": [1.0, 2.0]})
    test = pd.DataFrame({"x": [3.0], "c": pd.Categorical(["z"])})
    out = CategoryAligner().fit(train).transform(test)
    assert list(out.columns) == ["c", "x"]
    assert out["c"].isna().all()
    assert list(out["c"].cat.categories) == ["a", "b"]


QUICK = {
    "logreg": {"max_iter": 500},
    "random_forest": {"n_estimators": 30},
    "mlp": {"max_iter": 30},
    "xgboost": {"n_estimators": 30},
    "lightgbm": {"n_estimators": 30, "verbose": -1},
    "catboost": {"iterations": 30, "verbose": 0},
}
NEEDS = {"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost"}


@pytest.mark.parametrize("name", list(QUICK))
def test_fit_and_score_with_unseen_category(name, baf_frame):
    if name in NEEDS:
        pytest.importorskip(NEEDS[name])
    split = temporal_split(baf_frame, train=range(6), test=[6, 7])
    X_train, y_train = split.xy("train")
    X_test, _ = split.xy("test")
    X_test = X_test.copy()
    X_test["device_os"] = X_test["device_os"].cat.add_categories("brand_new")
    X_test.loc[X_test.index[:5], "device_os"] = "brand_new"

    model = build_model(name, QUICK[name], random_state=0).fit(X_train, y_train)
    s = predict_scores(model, X_test)
    assert s.shape == (len(X_test),)
    assert np.all((s >= 0) & (s <= 1))


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "params"),
    [("tabm", {"n_epochs": 2, "device": "cpu"}), ("realmlp", {"n_epochs": 2, "device": "cpu"})],
)
def test_deep_learning_wrappers(name, params, baf_frame):
    pytest.importorskip("pytabkit")
    split = temporal_split(baf_frame, train=range(6), test=[6, 7])
    X_train, y_train = split.xy("train")
    X_test, _ = split.xy("test")
    model = build_model(name, params, random_state=0).fit(X_train, y_train)
    assert predict_scores(model, X_test).shape == (len(X_test),)


def test_sample_params_respects_bounds():
    space = load_yaml(project_root() / "configs" / "models" / "lightgbm_baf_space.yaml")["space"]
    rng = np.random.default_rng(0)
    for _ in range(200):
        params = sample_params(space, rng)
        assert 20 <= params["n_estimators"] <= 10000
        assert 0.02 <= params["learning_rate"] <= 0.1
        assert params["boosting_type"] in {"gbdt", "goss"}
        assert isinstance(params["enable_bundle"], bool)


def test_goss_mapping_for_lightgbm4():
    out = to_lightgbm_params({"boosting_type": "goss", "num_leaves": 31})
    assert out == {"boosting_type": "gbdt", "data_sample_strategy": "goss", "num_leaves": 31}


def test_cli_lists_models(capsys):
    assert main(["models"]) == 0
    assert "tabpfn" in capsys.readouterr().out


def test_cli_smoke_runs_selected_models(capsys):
    assert main(["smoke", "--cpu", "--models", "logreg", "random_forest"]) == 0
    out = capsys.readouterr().out
    assert "[OK] logreg" in out and "2/2 models OK" in out


def test_smoke_reports_failures_instead_of_raising():
    from frauddetect.smoke import run_smoke

    (result,) = run_smoke(["not_a_model"])
    assert not result.ok and "KeyError" in result.detail
