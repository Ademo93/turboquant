"""Benchmark — latency, memory, size, accuracy.

The public entry point is :func:`compare`, which runs the same probes against
a baseline and a candidate model and returns a comparable :class:`BenchReport`.
"""

from turboquant.benchmark.accuracy import perplexity, top_k_accuracy
from turboquant.benchmark.compare import BenchReport, compare
from turboquant.benchmark.latency import LatencyResult, measure_latency
from turboquant.benchmark.memory import measure_memory, model_size_bytes

__all__ = [
    "BenchReport",
    "LatencyResult",
    "compare",
    "measure_latency",
    "measure_memory",
    "model_size_bytes",
    "perplexity",
    "top_k_accuracy",
]
