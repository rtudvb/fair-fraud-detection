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
    # Early stopping: hold out this share of each training set (stratified) and keep the
    # weights with the lowest held-out loss; stop after ``patience`` rounds without a gain.
    # None (the default) trains for the full budget, as in the first version of E10.
    val_fraction: float | None = None
    patience: int = 5


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


def split_holdout(X: np.ndarray, y: np.ndarray, fraction: float, rng):
    """Stratified (train, held-out) split; every class keeps at least one training row."""
    y = np.asarray(y)
    held = np.zeros(len(y), dtype=bool)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        n = min(round(fraction * idx.size), idx.size - 1)
        held[rng.choice(idx, size=max(n, 0), replace=False)] = True
    return (X[~held], y[~held]), (X[held], y[held])


def _holdout_loss(model, held: Sequence[tuple[np.ndarray, np.ndarray]], cfg: TrainConfig):
    """Size-weighted BCE (with the training ``pos_weight``) over the held-out sets."""
    import torch

    model.eval()
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg.pos_weight], device=cfg.device), reduction="sum"
    )
    total, count = 0.0, 0
    with torch.no_grad():
        for X, y in held:
            if len(y) == 0:
                continue
            logits = model(torch.from_numpy(X).to(cfg.device)).squeeze(1)
            target = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(cfg.device)
            total += float(loss_fn(logits, target))
            count += len(y)
    return total / max(count, 1)


class _EarlyStopping:
    """Keeps the best weights seen so far; ``step`` returns True when training should stop."""

    def __init__(self, patience: int) -> None:
        self.patience, self.best, self.state, self.waited = patience, np.inf, None, 0

    def step(self, model, loss: float) -> bool:
        if loss < self.best:
            self.best, self.state, self.waited = loss, copy.deepcopy(model.state_dict()), 0
            return False
        self.waited += 1
        return self.waited >= self.patience

    def restore(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)
        return model


def train_single(X: np.ndarray, y: np.ndarray, cfg: TrainConfig):
    """Ordinary training on one data set (a local bank, or pooled data as reference).

    Uses up to ``rounds * local_epochs`` epochs, the same budget as federated training; with
    ``cfg.val_fraction`` set, stops early on the held-out loss and keeps the best weights.
    """
    import torch

    torch.manual_seed(cfg.seed)
    model = _mlp(X.shape[1], cfg.hidden).to(cfg.device)
    rng = np.random.default_rng(cfg.seed)
    if not cfg.val_fraction:
        return _train_epochs(model, X, y, cfg.rounds * cfg.local_epochs, cfg, rng)
    (X_fit, y_fit), held = split_holdout(X, y, cfg.val_fraction, rng)
    stopper = _EarlyStopping(cfg.patience)
    for _ in range(cfg.rounds * cfg.local_epochs):
        _train_epochs(model, X_fit, y_fit, 1, cfg, rng)
        if stopper.step(model, _holdout_loss(model, [held], cfg)):
            break
    return stopper.restore(model)


def train_fedavg(clients: Sequence[tuple[np.ndarray, np.ndarray]], cfg: TrainConfig):
    """FedAvg: each round, every bank trains locally from the global model; average weights."""
    import torch

    torch.manual_seed(cfg.seed)
    global_model = _mlp(clients[0][0].shape[1], cfg.hidden).to(cfg.device)
    sizes = np.array([len(X) for X, _ in clients], dtype=float)
    weights = sizes / sizes.sum()
    rng = np.random.default_rng(cfg.seed)
    held: list[tuple[np.ndarray, np.ndarray]] = []
    if cfg.val_fraction:  # each bank keeps its own held-out rows; only the loss is shared
        splits = [split_holdout(X, y, cfg.val_fraction, rng) for X, y in clients]
        clients = [fit for fit, _ in splits]
        held = [h for _, h in splits]
    stopper = _EarlyStopping(cfg.patience)
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
        if held and stopper.step(global_model, _holdout_loss(global_model, held, cfg)):
            break
    return stopper.restore(global_model) if held else global_model
