"""Load YAML experiment configs and resolve repository paths."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def project_root(start: Path | None = None) -> Path:
    """Return the repository root: the closest parent holding ``configs/base.yaml``.

    Search order: ``FRAUDDETECT_ROOT``, then ``start`` (default: this file, which works for
    editable installs), then the current working directory (regular installs, notebooks).
    """
    if env := os.environ.get("FRAUDDETECT_ROOT"):
        return Path(env).resolve()
    for origin in (start or Path(__file__), Path.cwd()):
        here = origin.resolve()
        for candidate in (here, *here.parents):
            if (candidate / "configs" / "base.yaml").is_file():
                return candidate
    raise FileNotFoundError(
        "Could not locate the repository root; run from inside the repository "
        "or set FRAUDDETECT_ROOT."
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dict (an empty file yields an empty dict)."""
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level.")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(*paths: str | Path, root: Path | None = None) -> dict[str, Any]:
    """Load ``configs/base.yaml`` and deep-merge each extra config on top, in order.

    Relative paths are resolved against the repository root, and every entry under
    ``paths`` is turned into an absolute :class:`~pathlib.Path`.
    """
    root = root or project_root()
    config = load_yaml(root / "configs" / "base.yaml")
    for path in paths:
        path = Path(path)
        config = deep_merge(config, load_yaml(path if path.is_absolute() else root / path))
    config["paths"] = {key: root / value for key, value in config.get("paths", {}).items()}
    config["root"] = root
    return config
