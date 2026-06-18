"""FP16 / BF16 casting — the cheapest 2x compression you can apply.

These are not "real" quantization (no integer arithmetic), but they are the
universal baseline every quantization comparison should include.
"""

from __future__ import annotations

import torch
from torch import nn


def to_fp16(model: nn.Module, **_: object) -> nn.Module:
    """Cast all floating-point parameters and buffers to ``torch.float16``.

    Note
    ----
    FP16 can overflow during training. For activations that need a larger
    dynamic range (LayerNorm, softmax) consider BF16 instead.
    """
    return _cast(model, torch.float16)


def to_bf16(model: nn.Module, **_: object) -> nn.Module:
    """Cast to ``torch.bfloat16`` — same dynamic range as FP32, half the bytes."""
    return _cast(model, torch.bfloat16)


def _cast(model: nn.Module, dtype: torch.dtype) -> nn.Module:
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.to(dtype)
    for b in model.buffers():
        if b.is_floating_point():
            b.data = b.data.to(dtype)
    return model
