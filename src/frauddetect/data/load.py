"""Read BAF variants from ``data/raw`` (Parquet preferred, CSV as fallback)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from frauddetect.data.schema import CATEGORICAL_FEATURES, validate_frame


def variant_path(raw_dir: Path, stem: str) -> Path:
    """Return the file for a variant, preferring ``<stem>.parquet`` over ``<stem>.csv``."""
    for suffix in (".parquet", ".csv"):
        path = raw_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No {stem}.parquet or {stem}.csv in {raw_dir}. Run `make data` (see data/README.md)."
    )


def load_variant(config: dict, variant: str = "base", *, validate: bool = True) -> pd.DataFrame:
    """Load one BAF variant by its config key (e.g. ``"base"``, ``"variant_2"``).

    Categorical features are returned as pandas ``category`` dtype so every model wrapper
    receives the same, explicit encoding.
    """
    stems = config["dataset"]["variants"]
    if variant not in stems:
        raise KeyError(f"Unknown variant {variant!r}; choose from {sorted(stems)}")
    path = variant_path(config["paths"]["raw"], stems[variant])
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    if validate:
        validate_frame(df)
    return df
