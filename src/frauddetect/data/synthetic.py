"""Small BAF-shaped synthetic frames for tests, CI and smoke runs (no real data needed).

The values are random; only the schema and rough marginals (prevalence, age bins, months)
imitate BAF. Never use this for reported results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from frauddetect.data.schema import CATEGORICAL_FEATURES, LABEL, MONTH, NUMERIC_FEATURES

_CATEGORIES = {
    "payment_type": ["AA", "AB", "AC", "AD", "AE"],
    "employment_status": ["CA", "CB", "CC", "CD", "CE", "CF", "CG"],
    "housing_status": ["BA", "BB", "BC", "BD", "BE", "BF", "BG"],
    "source": ["INTERNET", "TELEAPP"],
    "device_os": ["linux", "macintosh", "other", "windows", "x11"],
}


def make_synthetic_baf(n: int = 5000, *, prevalence: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """Return ``n`` rows with the BAF columns and a label that depends on a few features.

    Older applicants get a higher base fraud rate, mirroring the prevalence disparity of
    the BAF Base variant, so fairness code has something to measure.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    for col in NUMERIC_FEATURES:
        data[col] = rng.normal(size=n)
    data["customer_age"] = rng.choice(np.arange(10, 100, 10), size=n, p=_age_weights())
    for col in (
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "has_other_cards",
        "foreign_request",
        "keep_alive_session",
    ):
        data[col] = rng.integers(0, 2, size=n)
    data["device_fraud_count"] = np.zeros(n, dtype=int)
    for col in CATEGORICAL_FEATURES:
        data[col] = rng.choice(_CATEGORIES[col], size=n)
    data[MONTH] = rng.integers(0, 8, size=n)

    older = data["customer_age"] >= 50
    logit = (
        np.log(prevalence / (1 - prevalence))
        + 1.2 * data["velocity_6h"]
        - 0.8 * data["name_email_similarity"]
        + 0.7 * older
        + 0.9 * (data["housing_status"] == "BA")
    )
    data[LABEL] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(data)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


def _age_weights() -> np.ndarray:
    weights = np.array([0.02, 0.22, 0.27, 0.22, 0.14, 0.08, 0.03, 0.015, 0.005])
    return weights / weights.sum()
