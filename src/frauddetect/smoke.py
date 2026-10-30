"""Environment check: GPU visibility and a tiny fit of every installed model.

Run ``python -m frauddetect smoke`` on a new machine before the real experiments. Each model
trains on a small synthetic BAF-shaped frame with minimal settings, so a pass means the
installation, the GPU and (for foundation models) the weight download all work.
"""

from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

from frauddetect.data.split import temporal_split
from frauddetect.data.synthetic import make_synthetic_baf
from frauddetect.models.registry import MODELS, build_model, predict_scores

# Minimal settings so every model finishes in seconds; not used for any reported result.
_QUICK_PARAMS: dict[str, dict] = {
    "logreg": {"max_iter": 500},
    "random_forest": {"n_estimators": 50, "n_jobs": -1},
    "xgboost": {"n_estimators": 50},
    "lightgbm": {"n_estimators": 50, "verbose": -1},
    "catboost": {"iterations": 50, "verbose": 0},
    "mlp": {"max_iter": 50},
    "tabm": {"n_epochs": 2},
    "realmlp": {"n_epochs": 2},
    "tabpfn": {},
    "tabicl": {},
}
_USES_TORCH_DEVICE = {"tabm", "realmlp", "tabpfn", "tabicl"}


@dataclass
class SmokeResult:
    model: str
    ok: bool
    seconds: float
    detail: str


def environment_report() -> tuple[list[str], str]:
    """Lines describing Python, PyTorch and the GPU, plus the torch device to use."""
    lines = [f"Python   {sys.version.split()[0]} ({platform.system()} {platform.release()})"]
    try:
        import torch
    except ImportError:
        lines.append("PyTorch  not installed (deep learning / foundation models unavailable)")
        return lines, "cpu"
    lines.append(f"PyTorch  {torch.__version__} (built for CUDA {torch.version.cuda})")
    if not torch.cuda.is_available():
        lines.append("GPU      NOT available -> models fall back to CPU (see docs/compute.md)")
        return lines, "cpu"
    props = torch.cuda.get_device_properties(0)
    lines.append(
        f"GPU      {props.name}, {props.total_memory / 1024**3:.1f} GB, "
        f"compute capability {props.major}.{props.minor}"
    )
    x = torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    lines.append(f"GPU test matrix multiply OK (checksum {float((x @ x).sum()):.1f})")
    return lines, "cuda"


def run_smoke(models: list[str] | None = None, *, device: str = "cpu") -> list[SmokeResult]:
    """Fit each model on synthetic data; never raises, failures are reported per model."""
    df = make_synthetic_baf(n=1200, prevalence=0.08, seed=0)
    split = temporal_split(df, train=range(6), test=[6, 7])
    X_train, y_train = split.xy("train")
    X_test, y_test = split.xy("test")

    results = []
    for name in models or list(MODELS):
        params = dict(_QUICK_PARAMS.get(name, {}))
        if name in _USES_TORCH_DEVICE:
            params["device"] = device
        start = time.perf_counter()
        try:
            model = build_model(name, params, random_state=0).fit(X_train, y_train)
            auc = roc_auc_score(y_test, predict_scores(model, X_test))
            results.append(SmokeResult(name, True, time.perf_counter() - start, f"AUC {auc:.3f}"))
        except Exception as exc:  # report and continue with the next model
            message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
            results.append(
                SmokeResult(
                    name, False, time.perf_counter() - start, f"{type(exc).__name__}: {message}"
                )
            )
    return results
