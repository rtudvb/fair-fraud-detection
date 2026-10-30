"""Federated averaging (McMahan et al., 2017) of an MLP across simulated banks.

Raw applications never leave a bank. What is shared:

* category levels and per-feature counts / sums / sums of squares, from which the exact
  global standardisation is computed (aggregates only, compatible with secure aggregation);
* model weights after each round, averaged in proportion to each bank's training size.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class FederatedPreprocessor:
    """One-hot categories + global standardisation built from client aggregates."""

    categories: dict[str, list] = field(default_factory=dict)
    means: dict[str, float] = field(default_factory=dict)
    stds: dict[str, float] = field(default_factory=dict)
    numeric: list[str] = field(default_factory=list)

    @classmethod
    def from_clients(cls, frames: Sequence[pd.DataFrame]) -> FederatedPreprocessor:
        first = frames[0]
        cat_cols = [c for c in first.columns if isinstance(first[c].dtype, pd.CategoricalDtype)]
        numeric = [c for c in first.columns if c not in cat_cols]
        categories = {
            c: sorted({v for f in frames for v in f[c].dropna().astype(str).unique()})
            for c in cat_cols
        }
        count = sum(len(f) for f in frames)
        sums = sum(f[numeric].to_numpy(dtype=float).sum(axis=0) for f in frames)
        squares = sum((f[numeric].to_numpy(dtype=float) ** 2).sum(axis=0) for f in frames)
        mean = sums / count
        std = np.sqrt(np.maximum(squares / count - mean**2, 0.0))
        std[std == 0] = 1.0
        return cls(
            categories=categories,
            means=dict(zip(numeric, mean, strict=True)),
            stds=dict(zip(numeric, std, strict=True)),
            numeric=numeric,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts = [
            (
                (frame[self.numeric].to_numpy(dtype=float) - np.array(list(self.means.values())))
                / np.array(list(self.stds.values()))
            )
        ]
        for col, levels in self.categories.items():
            values = frame[col].astype(str).to_numpy()
            parts.append((values[:, None] == np.array(levels)[None, :]).astype(float))
        return np.hstack(parts).astype(np.float32)


def _mlp(n_inputs: int, hidden: Sequence[int]):
    import torch.nn as nn

    layers, width = [], n_inputs
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU()]
        width = size
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


@dataclass
class TrainConfig:
    hidden: Sequence[int] = (128, 64)
    batch_size: int = 1024
    lr: float = 1e-3
    local_epochs: int = 1
    rounds: int = 30
    pos_weight: float = 1.0
    device: str = "cpu"
    seed: int = 0


def _train_epochs(model, X: np.ndarray, y: np.ndarray, epochs: int, cfg: TrainConfig, rng):
    import torch

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg.pos_weight], device=cfg.device)
    )
    X_t = torch.from_numpy(X).to(cfg.device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(cfg.device)
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(X))).to(cfg.device)
        for start in range(0, len(X), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_t[idx]).squeeze(1), y_t[idx])
            loss.backward()
            optimizer.step()
    return model


def predict(model, X: np.ndarray, device: str = "cpu", batch_size: int = 65536) -> np.ndarray:
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).to(device)
            out.append(torch.sigmoid(model(batch).squeeze(1)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def train_single(X: np.ndarray, y: np.ndarray, cfg: TrainConfig):
    """Ordinary training on one data set (a local bank, or pooled data as reference).

    Uses ``rounds * local_epochs`` epochs, the same amount of passes as federated training.
    """
    import torch

    torch.manual_seed(cfg.seed)
    model = _mlp(X.shape[1], cfg.hidden).to(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    return _train_epochs(model, X, y, cfg.rounds * cfg.local_epochs, cfg, rng)


def train_fedavg(clients: Sequence[tuple[np.ndarray, np.ndarray]], cfg: TrainConfig):
    """FedAvg: each round, every bank trains locally from the global model; average weights."""
    import torch

    torch.manual_seed(cfg.seed)
    global_model = _mlp(clients[0][0].shape[1], cfg.hidden).to(cfg.device)
    sizes = np.array([len(X) for X, _ in clients], dtype=float)
    weights = sizes / sizes.sum()
    rng = np.random.default_rng(cfg.seed)
    for round_ in range(cfg.rounds):
        states = []
        for X, y in clients:
            local = copy.deepcopy(global_model)
            _train_epochs(local, X, y, cfg.local_epochs, cfg, rng)
            states.append(local.state_dict())
        averaged = {
            key: sum(w * state[key] for w, state in zip(weights, states, strict=True))
            for key in states[0]
        }
        global_model.load_state_dict(averaged)
        log.debug("FedAvg round %d/%d done", round_ + 1, cfg.rounds)
    return global_model
