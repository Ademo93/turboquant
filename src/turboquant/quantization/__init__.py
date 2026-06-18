"""Quantization dispatch.

The public entry point is :func:`quantize`, which routes to a concrete backend
based on the ``method`` argument. All backends share the same signature:

    quantize_xxx(model, **config) -> nn.Module

so the dispatcher is intentionally thin.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from turboquant.quantization import awq_int as awq_mod
from turboquant.quantization import bnb, dtype_cast, gptq_int, int8_dynamic, int8_static

Method = Literal[
    "fp16",
    "bf16",
    "int8-dynamic",
    "int8-static",
    "bnb-int8",
    "bnb-nf4",
    "bnb-fp4",
    "gptq",
    "awq",
]

_REGISTRY = {
    "fp16": dtype_cast.to_fp16,
    "bf16": dtype_cast.to_bf16,
    "int8-dynamic": int8_dynamic.quantize_int8_dynamic,
    "int8-static": int8_static.quantize_int8_static,
    "bnb-int8": bnb.quantize_bnb_int8,
    "bnb-nf4": bnb.quantize_bnb_nf4,
    "bnb-fp4": bnb.quantize_bnb_fp4,
    "gptq": gptq_int.quantize_gptq,
    "awq": awq_mod.quantize_awq,
}


@dataclass
class QuantConfig:
    """Carrier for backend-specific options.

    Unknown fields are forwarded as kwargs to the backend, so adding new options
    does not require touching the dispatcher.
    """

    method: Method
    bits: int | None = None
    group_size: int = 128
    sym: bool = True
    calib_dataset: str | None = None
    calib_samples: int = 128
    seq_len: int = 2048
    extras: dict[str, Any] = field(default_factory=dict)


def quantize(model: nn.Module, method: Method | str, **kwargs: Any) -> nn.Module:
    """Quantize ``model`` using the named backend.

    Examples
    --------
    >>> from turboquant import quantize
    >>> q = quantize(model, method="bnb-nf4")              # doctest: +SKIP
    >>> q = quantize(model, method="gptq", bits=4)         # doctest: +SKIP
    """
    if method not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown quantization method '{method}'. Available: {available}")
    fn = _REGISTRY[method]
    return fn(model, **kwargs)


def list_methods() -> list[str]:
    """Return the list of registered quantization method names."""
    return sorted(_REGISTRY)


def is_quantized(model: nn.Module) -> bool:
    """Heuristic: a model is "quantized" if any parameter is sub-FP16 or uses a quant module."""
    for p in model.parameters():
        if p.dtype in (torch.int8, torch.uint8, torch.qint8, torch.quint8):
            return True
    for m in model.modules():
        cls = type(m).__name__
        if "Quant" in cls or "8bit" in cls or "4bit" in cls or "Linear4bit" in cls:
            return True
    return False


__all__ = [
    "QuantConfig",
    "is_quantized",
    "list_methods",
    "quantize",
]
