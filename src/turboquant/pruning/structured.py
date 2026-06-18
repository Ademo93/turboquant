"""Structured pruning — remove whole channels / filters.

Unlike unstructured sparsity, structured pruning yields immediate FLOPs and
latency reductions because the resulting weight tensor is genuinely smaller.
For real deployment one usually re-instantiates a *narrower* model after
pruning; the implementation below applies channel masks in-place, which is
sufficient for benchmarking and for export through ONNX shape inference.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn.utils.prune as prune_utils
from torch import nn


def prune_l1_channel(
    model: nn.Module,
    *,
    sparsity: float = 0.3,
    target_layers: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear),
    **_: object,
) -> nn.Module:
    return _structured(model, sparsity=sparsity, n=1, target_layers=target_layers)


def prune_l2_channel(
    model: nn.Module,
    *,
    sparsity: float = 0.3,
    target_layers: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear),
    **_: object,
) -> nn.Module:
    return _structured(model, sparsity=sparsity, n=2, target_layers=target_layers)


def prune_random_channel(
    model: nn.Module,
    *,
    sparsity: float = 0.3,
    target_layers: tuple[type[nn.Module], ...] = (nn.Conv2d, nn.Linear),
    **_: object,
) -> nn.Module:
    """Useful baseline: structured sparsity with random channel selection."""
    for module in model.modules():
        if isinstance(module, target_layers):
            prune_utils.random_structured(module, name="weight", amount=sparsity, dim=0)
            with contextlib.suppress(ValueError):
                prune_utils.remove(module, "weight")
    return model


def _structured(
    model: nn.Module,
    *,
    sparsity: float,
    n: int,
    target_layers: tuple[type[nn.Module], ...],
) -> nn.Module:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    for module in model.modules():
        if isinstance(module, target_layers):
            # dim=0 = output channels / output features
            prune_utils.ln_structured(module, name="weight", amount=sparsity, n=n, dim=0)
            with contextlib.suppress(ValueError):
                prune_utils.remove(module, "weight")
    return model


@torch.no_grad()
def estimate_flops_saved(
    model: nn.Module,
    input_shape: tuple[int, ...],
    *,
    device: str = "cpu",
) -> float:
    """Rough FLOPs estimate ignoring biases/activations.

    Compares "logical" weight FLOPs against the same model with zero-channel
    rows removed. Useful as a sanity check after structured pruning.
    """

    def _conv_flops(m: nn.Conv2d, x: torch.Tensor) -> int:
        oh = x.shape[-2] // m.stride[0]
        ow = x.shape[-1] // m.stride[1]
        nonzero_out = (m.weight.abs().sum(dim=(1, 2, 3)) > 0).sum().item()
        return int(nonzero_out * m.in_channels * m.kernel_size[0] * m.kernel_size[1] * oh * ow)

    def _linear_flops(m: nn.Linear) -> int:
        nonzero_out = (m.weight.abs().sum(dim=1) > 0).sum().item()
        return int(nonzero_out * m.in_features)

    flops = 0

    def hook_factory(name: str, mod: nn.Module):
        def hook(_, inputs, _output):
            nonlocal flops
            if isinstance(mod, nn.Conv2d):
                flops += _conv_flops(mod, inputs[0])
            elif isinstance(mod, nn.Linear):
                flops += _linear_flops(mod)

        return hook

    handles = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handles.append(m.register_forward_hook(hook_factory(name, m)))

    model.to(device).eval()
    with torch.no_grad():
        model(torch.randn(*input_shape, device=device))

    for h in handles:
        h.remove()
    return float(flops)
