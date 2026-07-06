from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer is not None:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def doc_dir() -> Path:
    return project_root() / "doc"


def code_dir() -> Path:
    return project_root() / "code"


def results_dir() -> Path:
    return project_root() / "results"


def figures_dir() -> Path:
    return project_root() / "figures"


def artifacts_dir() -> Path:
    return code_dir() / "artifacts"


def set_torch_num_threads_from_env(default: int = 8) -> None:
    v = os.environ.get("TORCH_NUM_THREADS")
    if v is not None:
        try:
            torch.set_num_threads(int(v))
        except ValueError:
            pass
    else:
        torch.set_num_threads(default)
