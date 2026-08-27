"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, num_threads: int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if num_threads:
        torch.set_num_threads(num_threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
