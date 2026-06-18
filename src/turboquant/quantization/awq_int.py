"""AWQ — Activation-aware Weight Quantization.

AWQ observes that the salience of LLM weights is dominated by a tiny fraction
of activation channels. By scaling those channels up before quantization (and
compensating by scaling their corresponding weights down), it preserves the
information that would otherwise be lost to rounding.

Reference: Lin et al., 2023 (arXiv:2306.00978).

This module wraps `autoawq`. It accepts a HuggingFace model id and writes a
quantized checkpoint that can be loaded back with `AutoAWQForCausalLM`.
"""

from __future__ import annotations

import importlib
from typing import Any

from torch import nn


def quantize_awq(
    model: nn.Module | str,
    *,
    bits: int = 4,
    group_size: int = 128,
    zero_point: bool = True,
    version: str = "GEMM",
    save_dir: str | None = None,
    tokenizer: Any | None = None,
    calib_dataset: str = "pileval",
    **_: object,
) -> nn.Module:
    """Quantize a causal LM with AWQ.

    Parameters
    ----------
    bits:
        Only 4-bit is well-supported by autoawq today.
    version:
        ``"GEMM"`` (default, fastest) or ``"GEMV"`` (slightly more accurate).
    """
    autoawq = _require("awq")
    transformers = _require("transformers")
    AutoAWQForCausalLM = autoawq.AutoAWQForCausalLM

    if not isinstance(model, str):
        raise NotImplementedError(
            "AWQ currently requires a model id or path; pass `model='org/model'`."
        )

    awq_model = AutoAWQForCausalLM.from_pretrained(model, safetensors=True)
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    quant_config = {
        "zero_point": zero_point,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": version,
    }
    awq_model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_dataset)

    if save_dir is not None:
        awq_model.save_quantized(save_dir)
        tokenizer.save_pretrained(save_dir)

    return awq_model


def _require(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"{name} is required for AWQ. Install with `pip install turboquant[awq]`."
        ) from e
