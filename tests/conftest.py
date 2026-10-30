import numpy as np
import pytest

from frauddetect.data.synthetic import make_synthetic_baf


@pytest.fixture(scope="session")
def baf_frame():
    """BAF-shaped synthetic frame (schema only; values are random)."""
    return make_synthetic_baf(n=3000, prevalence=0.05, seed=7)


@pytest.fixture
def scores():
    """Imbalanced labels with informative, tie-free scores."""
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.03).astype(int)
    s = rng.normal(loc=1.5 * y, scale=1.0)
    return y, s
