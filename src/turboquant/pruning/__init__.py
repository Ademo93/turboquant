"""Pruning dispatch.

Strategies are split between unstructured (sparse weights, requires sparse
kernels to pay off in latency) and structured (whole channels / filters
removed, immediately reduces FLOPs).
"""

from __future__ import annotations

from typing import Any, Literal

from torch import nn

from turboquant.pruning import magnitude, nm_sparsity, structured

Strategy = Literal[
    "magnitude",
    "l1-channel",
    "l2-channel",
    "random-channel",
    "nm-sparsity",
]

_REGISTRY = {
    "magnitude": magnitude.prune_magnitude,
    "l1-channel": structured.prune_l1_channel,
    "l2-channel": structured.prune_l2_channel,
    "random-channel": structured.prune_random_channel,
    "nm-sparsity": nm_sparsity.apply_nm_sparsity,
}


def prune(model: nn.Module, strategy: Strategy | str, **kwargs: Any) -> nn.Module:
    """Apply ``strategy`` to ``model`` and return it (in-place)."""
    if strategy not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown pruning strategy '{strategy}'. Available: {available}")
    return _REGISTRY[strategy](model, **kwargs)


def list_strategies() -> list[str]:
    return sorted(_REGISTRY)


def sparsity(model: nn.Module) -> float:
    """Fraction of zero-valued parameters across all floating-point tensors."""
    total = 0
    zero = 0
    for p in model.parameters():
        if not p.is_floating_point():
            continue
        total += p.numel()
        zero += (p == 0).sum().item()
    return zero / total if total else 0.0


__all__ = ["list_strategies", "prune", "sparsity"]
