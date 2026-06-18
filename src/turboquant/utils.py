"""Small cross-cutting helpers."""

from __future__ import annotations

import logging
import os
import random
from contextlib import contextmanager

import numpy as np
import torch

logger = logging.getLogger("turboquant")


def seed_everything(seed: int = 0) -> None:
    """Make a run deterministic across numpy / torch / cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


@contextmanager
def no_grad_inference_mode():
    """Best-effort: prefer ``inference_mode`` on torch >=1.9, else ``no_grad``."""
    if hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            yield
    else:
        with torch.no_grad():
            yield
