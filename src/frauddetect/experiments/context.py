"""Shared plumbing for every experiment: config, output folders, data access, metadata.

Real runs write to ``results/<experiment>/`` plus ``reports/paper/{figures,tables}``.
``--synthetic`` runs use generated BAF-shaped data and write everything under
``results/_synthetic/`` so pipeline checks can never be mistaken for reported results.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from frauddetect.config import deep_merge, load_config, load_yaml
from frauddetect.data.load import load_variant
from frauddetect.data.synthetic import make_synthetic_baf

log = logging.getLogger(__name__)

TORCH_MODELS = frozenset({"tabm", "realmlp", "tabpfn", "tabicl"})


def resolve_device(requested: str = "auto") -> str:
    """``auto`` -> ``cuda:0`` when PyTorch sees a GPU, else ``cpu``."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentContext:
    """Everything an experiment needs; build it with :func:`make_context`."""

    name: str
    config: dict[str, Any]
    synthetic: bool = False
    quick: bool = False
    models_override: list[str] | None = None
    device: str = "cpu"
    started: float = field(default_factory=time.time)
    _data: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ folders
    @property
    def root(self) -> Path:
        return self.config["root"]

    @property
    def results_root(self) -> Path:
        base = self.config["paths"]["results"]
        return base / "_synthetic" if self.synthetic else base

    @property
    def out_dir(self) -> Path:
        path = self.results_root / self.name
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def figures_dir(self) -> Path:
        path = self.results_root / "figures" if self.synthetic else self.config["paths"]["figures"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tables_dir(self) -> Path:
        path = self.results_root / "tables" if self.synthetic else self.config["paths"]["tables"]
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def scores_dir(self) -> Path:
        path = self.results_root / "scores"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------ config
    def section(self, key: str) -> dict[str, Any]:
        """Experiment section of ``configs/experiments.yaml`` (empty dict if absent)."""
        return dict(self.config["experiments"].get(key, {}))

    def models(self, key: str, field_name: str = "models") -> list[str]:
        return list(self.models_override or self.section(key).get(field_name, []))

    @property
    def seed(self) -> int:
        return int(self.config.get("seed", 0))

    @property
    def target_fpr(self) -> float:
        return float(self.config["evaluation"]["target_fpr"])

    def max_train_rows(self, model: str) -> int | None:
        return self.config["experiments"].get("max_train_rows", {}).get(model)

    # ------------------------------------------------------------------ data
    def load(self, variant: str = "base", *, cache: bool = True) -> pd.DataFrame:
        """A BAF variant, or a synthetic stand-in of the same schema with ``--synthetic``.

        Frames are cached per context, so several steps of one run read the data once.
        """
        if variant in self._data:
            return self._data[variant]
        if self.synthetic:
            rows = int(self.config["experiments"]["synthetic"]["rows"])
            keys = list(self.config["dataset"]["variants"])
            seed = 1000 + (keys.index(variant) if variant in keys else 0)
            df = make_synthetic_baf(n=rows, prevalence=0.03, seed=seed)
        else:
            df = load_variant(self.config, variant)
        if cache:
            self._data[variant] = df
        return df

    # ------------------------------------------------------------------ outputs
    def save_table(self, df: pd.DataFrame, name: str, *, floatfmt: str = ".4f") -> Path:
        """Full-precision CSV in the results folder and a Markdown copy for the report."""
        csv_path = self.out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        (self.tables_dir / f"{name}.md").write_text(to_markdown(df, floatfmt), encoding="utf-8")
        log.info("table  %s", csv_path)
        return csv_path

    def save_figure(self, fig, name: str) -> list[Path]:
        import matplotlib.pyplot as plt

        from frauddetect.viz import save_figure

        paths = save_figure(fig, name, self.figures_dir)
        plt.close(fig)
        log.info("figure %s", paths[0])
        return paths

    def save_json(self, payload: dict[str, Any], name: str) -> Path:
        path = self.out_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path

    def write_meta(self, **extra: Any) -> Path:
        """Record hardware, versions, git commit and the effective config of this run."""
        meta = {
            "experiment": self.name,
            "synthetic": self.synthetic,
            "quick": self.quick,
            "started": datetime.fromtimestamp(self.started, tz=timezone.utc).isoformat(),
            "seconds": round(time.time() - self.started, 1),
            **hardware_info(self.device),
            "git_commit": _git_commit(self.root),
            **extra,
        }
        (self.out_dir / "config.yaml").write_text(
            yaml.safe_dump(_plain(self.config), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return self.save_json(meta, "meta")


def make_context(
    name: str,
    *,
    synthetic: bool = False,
    quick: bool = False,
    models: list[str] | None = None,
    device: str | None = None,
    extra_configs: list[str] | None = None,
) -> ExperimentContext:
    """Load ``base.yaml`` + ``experiments.yaml`` (+ ``quick`` block) + extra configs."""
    config = load_config(*(extra_configs or []))
    experiments = load_yaml(config["root"] / "configs" / "experiments.yaml")
    quick_block = experiments.pop("quick", {})
    if quick:
        experiments = deep_merge(experiments, quick_block)
    config["experiments"] = experiments
    runtime = experiments.get("runtime", {})
    return ExperimentContext(
        name=name,
        config=config,
        synthetic=synthetic,
        quick=quick,
        models_override=models,
        device=resolve_device(device or runtime.get("device", "auto")),
    )


def hardware_info(device: str = "cpu") -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "cpu_count": os.cpu_count(),
        "device": device,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Pipe table without extra dependencies (Quarto renders it as a normal table)."""

    def fmt(value: Any) -> str:
        if isinstance(value, float | np.floating):
            return "" if np.isnan(value) else format(value, floatfmt)
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "|".join(" --- " for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows]) + "\n"


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _plain(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")
