"""Unit tests for the pruning module."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from turboquant.pruning import list_strategies, prune, sparsity
from turboquant.pruning.magnitude import measure_layer_sparsity, prune_magnitude
from turboquant.pruning.nm_sparsity import apply_nm_sparsity
from turboquant.pruning.structured import prune_l1_channel


def test_list_strategies_has_core_set() -> None:
    s = list_strategies()
    for expected in ("magnitude", "l1-channel", "l2-channel", "nm-sparsity"):
        assert expected in s


def test_unknown_strategy_raises(tiny_mlp: nn.Module) -> None:
    with pytest.raises(ValueError):
        prune(tiny_mlp, strategy="invalid")


def test_magnitude_pruning_achieves_target_sparsity(tiny_mlp: nn.Module) -> None:
    target = 0.5
    pruned = prune_magnitude(tiny_mlp, sparsity=target, scope="global")
    actual = sparsity(pruned)
    assert abs(actual - target) < 0.05, f"target={target} actual={actual}"


def test_magnitude_pruning_layerwise(tiny_mlp: nn.Module) -> None:
    pruned = prune_magnitude(tiny_mlp, sparsity=0.4, scope="layerwise")
    per_layer = measure_layer_sparsity(pruned)
    linears = [v for k, v in per_layer.items() if "fc" in k]
    assert all(0.35 <= v <= 0.45 for v in linears)


def test_l1_channel_keeps_forward_shape(tiny_conv: nn.Module, image_input: torch.Tensor) -> None:
    pruned = prune_l1_channel(tiny_conv, sparsity=0.25)
    out = pruned(image_input)
    assert out.shape == (2, 10)


def test_nm_sparsity_24_pattern(tiny_mlp: nn.Module) -> None:
    pruned = apply_nm_sparsity(tiny_mlp, n=2, m=4)
    # Inspect one fc layer
    w = pruned.fc2.weight.data
    rows = w.view(w.shape[0], -1)
    groups = rows.shape[1] // 4
    chunk = rows[:, : groups * 4].view(w.shape[0], groups, 4)
    nonzero_per_group = (chunk != 0).sum(dim=-1)
    assert (nonzero_per_group <= 2).all()


def test_nm_sparsity_rejects_invalid_params(tiny_mlp: nn.Module) -> None:
    with pytest.raises(ValueError):
        apply_nm_sparsity(tiny_mlp, n=4, m=2)


def test_sparsity_function_zero_on_dense_model(tiny_mlp: nn.Module) -> None:
    assert sparsity(tiny_mlp) == pytest.approx(0.0, abs=1e-6)
