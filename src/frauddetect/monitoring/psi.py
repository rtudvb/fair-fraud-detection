"""Population Stability Index (PSI) for post-deployment drift monitoring.

PSI = sum((cur - ref) * ln(cur / ref)) over bins. Common reading: < 0.1 stable,
0.1-0.25 moderate shift, > 0.25 major shift.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _psi_from_shares(ref: np.ndarray, cur: np.ndarray, eps: float) -> float:
    ref = np.clip(ref, eps, None)
    cur = np.clip(cur, eps, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(
    reference: ArrayLike, current: ArrayLike, bins: int = 10, eps: float = 1e-6
) -> float:
    """PSI of a numeric feature with quantile bins taken from the reference sample."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:  # constant reference: compare "equal to it" vs "different"
        same_ref = np.mean(ref == ref[0])
        same_cur = np.mean(cur == ref[0])
        return _psi_from_shares(
            np.array([same_ref, 1 - same_ref]), np.array([same_cur, 1 - same_cur]), eps
        )
    inner = edges[1:-1]  # open-ended outer bins catch values outside the reference range
    ref_counts = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=inner.size + 1)
    cur_counts = np.bincount(np.searchsorted(inner, cur, side="right"), minlength=inner.size + 1)
    return _psi_from_shares(ref_counts / ref.size, cur_counts / cur.size, eps)


def psi_categorical(reference: ArrayLike, current: ArrayLike, eps: float = 1e-6) -> float:
    """PSI of a categorical feature over the union of observed levels."""
    ref = pd.Series(np.asarray(reference, dtype=object))
    cur = pd.Series(np.asarray(current, dtype=object))
    levels = sorted(set(ref.unique()) | set(cur.unique()), key=str)
    ref_share = ref.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    cur_share = cur.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy()
    return _psi_from_shares(ref_share, cur_share, eps)


def psi_table(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Iterable[str] | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """PSI per column, sorted from most to least shifted, with a stability label."""
    columns = list(columns) if columns is not None else list(reference.columns)
    rows = []
    for col in columns:
        is_cat = (
            isinstance(reference[col].dtype, pd.CategoricalDtype) or reference[col].dtype == object
        )
        value = (
            psi_categorical(reference[col], current[col])
            if is_cat
            else psi_numeric(reference[col], current[col], bins=bins)
        )
        rows.append({"feature": col, "psi": value})
    table = pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
    table["status"] = pd.cut(
        table["psi"], bins=[-np.inf, 0.1, 0.25, np.inf], labels=["stable", "moderate", "major"]
    )
    return table
