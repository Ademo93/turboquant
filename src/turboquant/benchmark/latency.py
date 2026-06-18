"""Latency measurement with warm-up and CUDA-aware timing."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class LatencyResult:
    """Aggregated latency statistics, all in milliseconds."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    n_iters: int
    device: str

    def throughput(self, batch_size: int = 1) -> float:
        """Items per second based on mean latency."""
        return batch_size * 1000.0 / self.mean_ms if self.mean_ms > 0 else 0.0


@torch.no_grad()
def measure_latency(
    fn: Callable[[], object],
    *,
    warmup: int = 10,
    iters: int = 50,
    device: str | None = None,
) -> LatencyResult:
    """Time ``fn`` over ``iters`` runs after ``warmup`` warm-up runs.

    On CUDA, uses cuda events for sub-ms accuracy and synchronizes around each
    call. On CPU, falls back to ``time.perf_counter``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    for _ in range(warmup):
        fn()
        if device == "cuda":
            torch.cuda.synchronize()

    samples: list[float] = []
    if device == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1000)

    samples.sort()
    return LatencyResult(
        mean_ms=statistics.fmean(samples),
        median_ms=statistics.median(samples),
        p95_ms=samples[int(0.95 * (len(samples) - 1))],
        p99_ms=samples[int(0.99 * (len(samples) - 1))],
        n_iters=iters,
        device=device,
    )
