"""Split applications across simulated banks of unequal size and customer mix (non-IID)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def bank_probabilities(
    levels: Sequence, shares: Sequence[float], alpha: float, seed: int
) -> pd.DataFrame:
    """P(bank | level of the partition feature).

    Each level draws a Dirichlet(alpha) preference over banks; multiplying by the target
    bank shares keeps big banks big while giving every bank its own customer mix.
    Smaller ``alpha`` means more different banks.
    """
    rng = np.random.default_rng(seed)
    shares = np.asarray(shares, dtype=float)
    rows = {}
    for level in levels:
        weights = shares * rng.dirichlet(np.full(len(shares), alpha))
        rows[level] = weights / weights.sum()
    return pd.DataFrame.from_dict(rows, orient="index", columns=range(len(shares)))


def assign_banks(values: pd.Series, probabilities: pd.DataFrame, seed: int) -> np.ndarray:
    """Bank index for every row, drawn from the row's level-specific probabilities."""
    rng = np.random.default_rng(seed)
    banks = np.empty(len(values), dtype=int)
    codes = values.astype(object).to_numpy()
    for level, probs in probabilities.iterrows():
        mask = codes == level
        banks[mask] = rng.choice(len(probs), size=int(mask.sum()), p=probs.to_numpy())
    unknown = ~np.isin(codes, probabilities.index.to_numpy())
    if unknown.any():  # levels unseen when the probabilities were drawn: use overall mix
        overall = probabilities.mean().to_numpy()
        banks[unknown] = rng.choice(
            len(overall), size=int(unknown.sum()), p=overall / overall.sum()
        )
    return banks
