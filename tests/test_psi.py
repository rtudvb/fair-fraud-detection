import numpy as np
import pandas as pd

from frauddetect.monitoring.psi import psi_categorical, psi_numeric, psi_table


def test_same_distribution_is_stable():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=20000), rng.normal(size=20000)) < 0.01


def test_shifted_distribution_is_major():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=20000), rng.normal(loc=1.0, size=20000)) > 0.25


def test_values_outside_reference_range_are_counted():
    assert psi_numeric(np.arange(100), np.arange(1000, 1100)) > 1


def test_constant_reference():
    assert psi_numeric(np.zeros(100), np.zeros(100)) < 1e-9
    assert psi_numeric(np.zeros(100), np.ones(100)) > 1


def test_categorical_psi_with_new_level():
    ref = ["a"] * 50 + ["b"] * 50
    assert psi_categorical(ref, ref) < 1e-9
    assert psi_categorical(ref, ["a"] * 50 + ["c"] * 50) > 1


def test_psi_table_sorted_with_status():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(size=5000), "c": pd.Categorical(["a", "b"] * 2500)})
    cur = pd.DataFrame({"x": rng.normal(loc=2, size=5000), "c": pd.Categorical(["a", "b"] * 2500)})
    table = psi_table(ref, cur)
    assert list(table["feature"]) == ["x", "c"]
    assert list(table["status"].astype(str)) == ["major", "stable"]
