"""Memory and on-disk size measurements."""

from __future__ import annotations

import contextlib
import gc
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import psutil
import torch
from torch import nn


@dataclass
class MemoryStats:
    """All values in bytes."""

    peak_gpu: int
    peak_cpu: int
    device: str


@contextlib.contextmanager
def measure_memory(device: str | None = None) -> Iterator[MemoryStats]:
    """Context manager that tracks peak memory used inside the ``with`` block."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    stats = MemoryStats(peak_gpu=0, peak_cpu=0, device=device)
    proc = psutil.Process(os.getpid())
    cpu_before = proc.memory_info().rss

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        yield stats
    finally:
        if device == "cuda":
            stats.peak_gpu = int(torch.cuda.max_memory_allocated())
        stats.peak_cpu = max(0, int(proc.memory_info().rss - cpu_before))


def model_size_bytes(model: nn.Module) -> int:
    """Size of the serialized state dict on disk, in bytes.

    Save → stat → delete. This counts real serialized bytes — useful when
    bitsandbytes / GPTQ layers store packed integer weights.
    """
    gc.collect()
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), path)
        return os.path.getsize(path)
    finally:
        with contextlib.suppress(OSError):
            os.remove(path)
