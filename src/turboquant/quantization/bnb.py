"""BitsAndBytes wrappers — INT8 / NF4 / FP4 weight-only LLM quantization.

bitsandbytes provides drop-in replacements for ``nn.Linear`` (``Linear8bitLt``,
``Linear4bit``) that store weights in 8-bit or 4-bit blocks and dequantize on
the fly. This module swaps the relevant linear layers in a HuggingFace-style
model and leaves everything else untouched.

References
----------
- LLM.int8() — Dettmers et al., 2022 (arXiv:2208.07339)
- QLoRA / NF4 — Dettmers et al., 2023 (arXiv:2305.14314)
"""

from __future__ import annotations

import importlib
from typing import Any

from torch import nn


def _require_bnb() -> Any:
    try:
        return importlib.import_module("bitsandbytes")
    except ImportError as e:  # pragma: no cover - hard to test without bnb
        raise ImportError(
            "bitsandbytes is required. Install with `pip install turboquant[bnb]`."
        ) from e


def quantize_bnb_int8(
    model: nn.Module,
    *,
    threshold: float = 6.0,
    skip_modules: tuple[str, ...] = ("lm_head",),
    **_: object,
) -> nn.Module:
    """Replace ``nn.Linear`` with ``bnb.nn.Linear8bitLt`` (LLM.int8()).

    The 8-bit kernel keeps a small high-precision "outlier" branch above
    ``threshold`` to preserve accuracy on the rare large activations that hurt
    naive INT8.
    """
    bnb = _require_bnb()
    Linear8bitLt = bnb.nn.Linear8bitLt
    return _replace_linears(
        model,
        skip=skip_modules,
        build=lambda lin: Linear8bitLt(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            has_fp16_weights=False,
            threshold=threshold,
        ),
        copy_weights=True,
    )


def quantize_bnb_nf4(
    model: nn.Module,
    *,
    compute_dtype: str = "float16",
    double_quant: bool = True,
    skip_modules: tuple[str, ...] = ("lm_head",),
    **_: object,
) -> nn.Module:
    """4-bit NormalFloat (NF4) quantization — the recipe used in QLoRA."""
    return _quantize_4bit(
        model,
        quant_type="nf4",
        compute_dtype=compute_dtype,
        double_quant=double_quant,
        skip_modules=skip_modules,
    )


def quantize_bnb_fp4(
    model: nn.Module,
    *,
    compute_dtype: str = "float16",
    double_quant: bool = True,
    skip_modules: tuple[str, ...] = ("lm_head",),
    **_: object,
) -> nn.Module:
    """4-bit floating-point quantization (FP4)."""
    return _quantize_4bit(
        model,
        quant_type="fp4",
        compute_dtype=compute_dtype,
        double_quant=double_quant,
        skip_modules=skip_modules,
    )


def _quantize_4bit(
    model: nn.Module,
    *,
    quant_type: str,
    compute_dtype: str,
    double_quant: bool,
    skip_modules: tuple[str, ...],
) -> nn.Module:
    import torch

    bnb = _require_bnb()
    Linear4bit = bnb.nn.Linear4bit
    dtype = getattr(torch, compute_dtype)
    return _replace_linears(
        model,
        skip=skip_modules,
        build=lambda lin: Linear4bit(
            lin.in_features,
            lin.out_features,
            bias=lin.bias is not None,
            compute_dtype=dtype,
            compress_statistics=double_quant,
            quant_type=quant_type,
        ),
        copy_weights=True,
    )


def _replace_linears(
    model: nn.Module,
    *,
    skip: tuple[str, ...],
    build,
    copy_weights: bool,
) -> nn.Module:
    """Walk the module tree and swap eligible ``nn.Linear`` layers."""
    for name, module in model.named_children():
        if any(s in name for s in skip):
            continue
        if isinstance(module, nn.Linear):
            new = build(module)
            if copy_weights:
                new.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    new.bias.data = module.bias.data.clone()
            setattr(model, name, new)
        else:
            _replace_linears(module, skip=skip, build=build, copy_weights=copy_weights)
    return model
