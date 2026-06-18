"""Unit tests for benchmark helpers."""

from __future__ import annotations

import torch
from torch import nn

from turboquant.benchmark import (
    measure_latency,
    measure_memory,
    model_size_bytes,
)
from turboquant.benchmark.compare import compare


def test_measure_latency_returns_reasonable_stats(
    tiny_mlp: nn.Module, mlp_input: torch.Tensor
) -> None:
    res = measure_latency(lambda: tiny_mlp(mlp_input), warmup=2, iters=5, device="cpu")
    assert res.n_iters == 5
    assert res.mean_ms > 0
    assert res.p95_ms >= res.median_ms


def test_model_size_bytes_positive(tiny_mlp: nn.Module) -> None:
    assert model_size_bytes(tiny_mlp) > 0


def test_measure_memory_context(tiny_mlp: nn.Module, mlp_input: torch.Tensor) -> None:
    with measure_memory(device="cpu") as mem:
        _ = tiny_mlp(mlp_input)
    assert mem.peak_cpu >= 0


def test_compare_produces_two_rows(tiny_mlp: nn.Module, mlp_input: torch.Tensor) -> None:
    rep = compare(
        tiny_mlp,
        tiny_mlp,
        sample_input=mlp_input,
        metrics=("latency", "memory", "size"),
        warmup=1,
        iters=3,
        device="cpu",
    )
    assert len(rep.runs) == 2
    for r in rep.runs:
        assert r.size_mb is not None
        assert r.latency_ms is not None


def test_compare_table_string_renders(tiny_mlp: nn.Module, mlp_input: torch.Tensor) -> None:
    rep = compare(tiny_mlp, tiny_mlp, sample_input=mlp_input, warmup=1, iters=2, device="cpu")
    s = rep.as_table()
    assert "name" in s
    assert "latency_ms" in s
