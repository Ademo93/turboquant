"""N:M structured sparsity (e.g. 2:4) — the format NVIDIA Ampere Tensor Cores accelerate.

Inside every contiguous group of M consecutive weights, only N are allowed to
be non-zero. The resulting pattern is hardware-friendly (no metadata needed for
2:4) and gives a real ~2x speedup on supported GPUs.
"""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def apply_nm_sparsity(
    model: nn.Module,
    *,
    n: int = 2,
    m: int = 4,
    target_layers: tuple[type[nn.Module], ...] = (nn.Linear, nn.Conv2d),
    **_: object,
) -> nn.Module:
    """Zero out (M-N) weights per group of M along the input dimension.

    The default (n=2, m=4) is the only pattern accelerated by Ampere & Hopper
    sparse Tensor Cores.
    """
    if n >= m or n <= 0 or m <= 0:
        raise ValueError("Require 0 < n < m")

    for module in model.modules():
        if not isinstance(module, target_layers):
            continue
        w = module.weight.data
        original_shape = w.shape
        # Group along the *input* dimension (last for Linear, dim=1 for Conv).
        w2d = w.view(original_shape[0], -1)
        groups = w2d.shape[1] // m
        if groups == 0:
            continue
        trimmed = w2d[:, : groups * m].view(original_shape[0], groups, m)
        # Keep the n largest-abs entries in each group.
        _, idx = trimmed.abs().topk(n, dim=-1)
        mask = torch.zeros_like(trimmed)
        mask.scatter_(-1, idx, 1.0)
        trimmed.mul_(mask)
        w2d[:, : groups * m] = trimmed.view(original_shape[0], groups * m)
        module.weight.data = w2d.view(original_shape)
    return model
