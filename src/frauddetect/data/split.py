"""Temporal train / validation / test splits on the ``month`` column."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from frauddetect.data.schema import LABEL, MONTH


@dataclass(frozen=True)
class Split:
    """Frames of one temporal split; ``valid`` is ``None`` when the protocol has none."""

    train: pd.DataFrame
    test: pd.DataFrame
    valid: pd.DataFrame | None = None

    def xy(self, part: str) -> tuple[pd.DataFrame, np.ndarray]:
        """Return features (without label and month) and the label array of one part."""
        frame = getattr(self, part)
        if frame is None:
            raise ValueError(f"This split has no {part!r} part.")
        return frame.drop(columns=[LABEL, MONTH]), frame[LABEL].to_numpy()


def temporal_split(
    df: pd.DataFrame,
    train: Sequence[int],
    test: Sequence[int],
    valid: Sequence[int] | None = None,
) -> Split:
    """Split ``df`` by month. Month sets must not overlap; row order is preserved."""
    parts = {"train": set(train), "test": set(test), "valid": set(valid or ())}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = parts[a] & parts[b]
            if overlap:
                raise ValueError(f"months {sorted(overlap)} appear in both {a} and {b}")

    def select(months: set[int]) -> pd.DataFrame:
        return df.loc[df[MONTH].isin(months)].reset_index(drop=True)

    return Split(
        train=select(parts["train"]),
        test=select(parts["test"]),
        valid=select(parts["valid"]) if valid else None,
    )


def split_from_config(df: pd.DataFrame, config: dict, protocol: str = "main") -> Split:
    """Apply a protocol from ``config['splits']`` (``"main"`` or ``"reproduction"``)."""
    spec = config["splits"][protocol]
    return temporal_split(df, train=spec["train"], test=spec["test"], valid=spec.get("valid"))


def stratified_subsample(
    df: pd.DataFrame, n: int | None, *, seed: int, label: str = LABEL
) -> pd.DataFrame:
    """Draw ``n`` rows keeping the label prevalence (for learning curves).

    ``None`` or ``n >= len(df)`` returns ``df`` unchanged. Every class keeps at least one row.
    """
    if n is None or n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    picks = []
    for _, group in df.groupby(label, sort=True):
        k = max(1, round(n * len(group) / len(df)))
        picks.append(rng.choice(group.index.to_numpy(), size=min(k, len(group)), replace=False))
    index = np.sort(np.concatenate(picks))
    return df.loc[index].reset_index(drop=True)
