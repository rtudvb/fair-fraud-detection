"""Small-bank dilemma (RQ6, experiment E10; go/no-go at the 11/15 checkpoint).

Five simulated banks of unequal size and customer mix compare local models (boosted trees,
a foundation model, an MLP), federated averaging of an MLP, and pooled training.
"""

from frauddetect.federated.fedavg import (
    FederatedPreprocessor,
    TrainConfig,
    predict,
    train_fedavg,
    train_single,
)
from frauddetect.federated.partition import assign_banks, bank_probabilities

__all__ = [
    "FederatedPreprocessor",
    "TrainConfig",
    "assign_banks",
    "bank_probabilities",
    "predict",
    "train_fedavg",
    "train_single",
]
