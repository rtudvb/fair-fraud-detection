import numpy as np
import pandas as pd
import pytest

from frauddetect.federated.partition import assign_banks, bank_probabilities


def test_bank_probabilities_rows_sum_to_one():
    probs = bank_probabilities(["a", "b", "c"], [0.5, 0.3, 0.2], alpha=0.5, seed=0)
    assert probs.shape == (3, 3)
    assert np.allclose(probs.sum(axis=1), 1)


def test_assign_banks_keeps_big_banks_big_and_handles_new_levels():
    probs = bank_probabilities(list("abcdefgh"), [0.6, 0.3, 0.1], alpha=5.0, seed=1)
    values = pd.Series(np.random.default_rng(0).choice(list("abcdefgh"), size=20000))
    banks = assign_banks(values, probs, seed=0)
    counts = np.bincount(banks, minlength=3) / len(banks)
    assert counts[0] > counts[1] > counts[2]
    assert set(assign_banks(pd.Series(["zzz"] * 10), probs, seed=0)) <= {0, 1, 2}


def test_federated_preprocessor_matches_pooled_statistics():
    from frauddetect.federated.fedavg import FederatedPreprocessor

    rng = np.random.default_rng(0)
    frames = [
        pd.DataFrame({"x": rng.normal(loc=i, size=100), "c": pd.Categorical(["u", "v"] * 50)})
        for i in range(3)
    ]
    prep = FederatedPreprocessor.from_clients(frames)
    pooled = pd.concat(frames)["x"]
    assert prep.means["x"] == pytest.approx(pooled.mean())
    assert prep.stds["x"] == pytest.approx(pooled.std(ddof=0))
    X = prep.transform(frames[0])
    assert X.shape == (100, 3)  # x + one-hot u, v
    assert X.dtype == np.float32


def test_fedavg_trains_and_predicts():
    pytest.importorskip("torch")
    from frauddetect.federated.fedavg import TrainConfig, predict, train_fedavg, train_single

    rng = np.random.default_rng(0)
    clients = []
    for _ in range(3):
        X = rng.normal(size=(300, 4)).astype(np.float32)
        y = (X[:, 0] + 0.5 * rng.normal(size=300) > 1).astype(int)
        clients.append((X, y))
    cfg = TrainConfig(hidden=(8,), rounds=5, batch_size=64, lr=0.01)
    model = train_fedavg(clients, cfg)
    scores = predict(model, clients[0][0])
    assert scores.shape == (300,)
    assert ((scores >= 0) & (scores <= 1)).all()
    y0 = clients[0][1].astype(bool)
    assert scores[y0].mean() > scores[~y0].mean()  # learned the signal
    local = train_single(*clients[0], cfg)
    assert predict(local, clients[0][0]).shape == (300,)


def test_early_stopping_keeps_best_weights():
    pytest.importorskip("torch")
    from frauddetect.federated.fedavg import (
        TrainConfig,
        predict,
        split_holdout,
        train_fedavg,
        train_single,
    )

    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 4)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=400) > 1).astype(int)
    (_, y_fit), (_, y_held) = split_holdout(X, y, 0.25, rng)
    assert len(y_fit) + len(y_held) == 400
    assert y_held.sum() >= 1 and y_fit.sum() >= 1  # both classes on both sides
    cfg = TrainConfig(hidden=(8,), rounds=40, batch_size=64, lr=0.01, val_fraction=0.2, patience=2)
    for model in (train_single(X, y, cfg), train_fedavg([(X, y), (X[:200], y[:200])], cfg)):
        scores = predict(model, X)
        assert scores[y == 1].mean() > scores[y == 0].mean()
