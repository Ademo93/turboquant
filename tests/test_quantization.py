"""Unit tests for quantization paths that don't need external weights."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from turboquant.quantization import dtype_cast, int8_dynamic, list_methods, observers, quantize


def test_list_methods_includes_core_set() -> None:
    methods = list_methods()
    for expected in ("fp16", "bf16", "int8-dynamic", "int8-static", "bnb-nf4", "gptq", "awq"):
        assert expected in methods


def test_quantize_unknown_method_raises(tiny_mlp: nn.Module) -> None:
    with pytest.raises(ValueError):
        quantize(tiny_mlp, method="does-not-exist")


def test_fp16_cast_changes_dtype(tiny_mlp: nn.Module) -> None:
    q = dtype_cast.to_fp16(tiny_mlp)
    assert all(p.dtype == torch.float16 for p in q.parameters() if p.is_floating_point())


def test_bf16_cast_changes_dtype(tiny_mlp: nn.Module) -> None:
    q = dtype_cast.to_bf16(tiny_mlp)
    assert all(p.dtype == torch.bfloat16 for p in q.parameters() if p.is_floating_point())


def test_int8_dynamic_runs_forward(tiny_mlp: nn.Module, mlp_input: torch.Tensor) -> None:
    q = int8_dynamic.quantize_int8_dynamic(tiny_mlp)
    out = q(mlp_input)
    assert out.shape == (4, 10)


def test_int8_dynamic_reduces_serialized_size(tiny_mlp: nn.Module) -> None:
    from turboquant.benchmark import model_size_bytes

    base_size = model_size_bytes(tiny_mlp)
    q = int8_dynamic.quantize_int8_dynamic(tiny_mlp)
    q_size = model_size_bytes(q)
    # INT8 dynamic should be meaningfully smaller; allow some headroom.
    assert q_size < base_size, f"INT8 ({q_size}) not smaller than FP32 ({base_size})"


def test_minmax_observer_symmetric_zero_point() -> None:
    obs = observers.MinMaxObserver(symmetric=True)
    obs.observe(torch.tensor([-1.0, 0.5]))
    obs.observe(torch.tensor([2.0, -0.2]))
    qp = obs.compute()
    assert qp.zero_point.item() == 0
    assert qp.scale.item() > 0


def test_minmax_observer_round_trip_within_tolerance() -> None:
    obs = observers.MinMaxObserver(symmetric=False)
    x = torch.linspace(-3, 5, 100)
    obs.observe(x)
    qp = obs.compute()
    recovered = qp.dequantize(qp.quantize(x))
    assert (recovered - x).abs().max() < (qp.scale.item() * 1.1)


def test_percentile_observer_ignores_outliers() -> None:
    obs = observers.PercentileObserver(percentile=99.0, symmetric=False)
    x = torch.cat([torch.randn(1000), torch.tensor([1000.0])])
    obs.observe(x)
    qp = obs.compute()
    assert qp.scale.item() < 100  # outlier did not blow up the scale
