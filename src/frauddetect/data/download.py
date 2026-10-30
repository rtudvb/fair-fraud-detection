"""Download the BAF suite from Kaggle and convert it to Parquet.

The data is licensed CC BY-NC-ND 4.0: it is downloaded locally and never committed.
Kaggle credentials are read from ``~/.kaggle/kaggle.json`` or the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables. On Kaggle Notebooks, attach the dataset instead and
point ``paths.raw`` at ``/kaggle/input/bank-account-fraud-dataset-neurips-2022``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def download_from_kaggle(slug: str, raw_dir: Path) -> None:
    """Download and unzip the Kaggle dataset ``slug`` into ``raw_dir``."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "The Kaggle client is not installed: `pip install -e '.[data]'`, "
            "or download the files manually (see data/README.md)."
        ) from exc
    raw_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    log.info("Downloading %s into %s", slug, raw_dir)
    api.dataset_download_files(slug, path=str(raw_dir), unzip=True, quiet=False)


def convert_csv_to_parquet(raw_dir: Path, *, remove_csv: bool = False) -> list[Path]:
    """Convert every ``*.csv`` in ``raw_dir`` to Parquet (about 3x smaller, much faster to read)."""
    written = []
    for csv_path in sorted(raw_dir.glob("*.csv")):
        parquet_path = csv_path.with_suffix(".parquet")
        if not parquet_path.exists():
            log.info("Converting %s", csv_path.name)
            pd.read_csv(csv_path).to_parquet(parquet_path, index=False)
        written.append(parquet_path)
        if remove_csv:
            csv_path.unlink()
    return written
