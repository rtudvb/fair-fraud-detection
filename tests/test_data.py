from pathlib import Path

import pytest

from frauddetect.config import deep_merge, load_config, project_root
from frauddetect.data.load import load_variant
from frauddetect.data.schema import FEATURES, LABEL, MONTH, validate_frame
from frauddetect.data.split import split_from_config, stratified_subsample, temporal_split


def test_schema_has_thirty_features():
    assert len(FEATURES) == 30


def test_synthetic_frame_is_valid(baf_frame):
    report = validate_frame(baf_frame)
    assert report.ok and not report.warnings


def test_validation_errors(baf_frame):
    bad = baf_frame.drop(columns=["income"]).assign(fraud_bool=2)
    report = validate_frame(bad, strict=False)
    assert len(report.errors) == 2
    with pytest.raises(ValueError):
        validate_frame(bad)


def test_extra_columns_only_warn(baf_frame):
    report = validate_frame(baf_frame.assign(x1=0.0), strict=False)
    assert report.ok and report.warnings


def test_temporal_split_months(baf_frame):
    split = temporal_split(baf_frame, train=[0, 1, 2, 3, 4], valid=[5], test=[6, 7])
    assert set(split.train[MONTH]) == {0, 1, 2, 3, 4}
    assert set(split.valid[MONTH]) == {5}
    assert set(split.test[MONTH]) == {6, 7}
    X, y = split.xy("test")
    assert LABEL not in X.columns and MONTH not in X.columns
    assert len(X) == len(y)


def test_overlapping_months_raise(baf_frame):
    with pytest.raises(ValueError):
        temporal_split(baf_frame, train=[0, 1], test=[1, 2])


def test_reproduction_protocol_has_no_validation(baf_frame):
    config = load_config()
    split = split_from_config(baf_frame, config, "reproduction")
    assert split.valid is None
    with pytest.raises(ValueError):
        split.xy("valid")


def test_stratified_subsample_keeps_prevalence(baf_frame):
    sample = stratified_subsample(baf_frame, 1000, seed=0)
    assert abs(len(sample) - 1000) <= 2
    assert sample[LABEL].mean() == pytest.approx(baf_frame[LABEL].mean(), abs=0.01)
    assert stratified_subsample(baf_frame, None, seed=0) is baf_frame


def test_load_variant_from_parquet(tmp_path: Path, baf_frame):
    baf_frame.to_parquet(tmp_path / "Base.parquet")
    config = load_config()
    config["paths"]["raw"] = tmp_path
    df = load_variant(config, "base")
    assert df.shape == baf_frame.shape
    assert str(df["device_os"].dtype) == "category"
    with pytest.raises(FileNotFoundError):
        load_variant(config, "variant_1")
    with pytest.raises(KeyError):
        load_variant(config, "nope")


def test_config_merging_and_paths():
    config = load_config()
    assert config["root"] == project_root()
    assert config["paths"]["raw"].is_absolute()
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}
