"""GPTQ — accurate post-training weight quantization for LLMs.

GPTQ uses approximate second-order information (the Hessian of the
layer-wise reconstruction loss) to quantize one column at a time while
compensating the remaining columns. For 4-bit weight-only quantization it
typically loses <0.1 perplexity on Llama-class models.

Reference: Frantar et al., 2023 (arXiv:2210.17323).

This module wraps `auto-gptq` to keep the implementation maintainable. When
``auto-gptq`` is not installed it raises a clear error rather than failing at
import time.
"""

from __future__ import annotations

import importlib
from typing import Any

from torch import nn


def quantize_gptq(
    model: nn.Module | str,
    *,
    bits: int = 4,
    group_size: int = 128,
    desc_act: bool = False,
    sym: bool = True,
    calib_dataset: str | list[str] = "wikitext",
    calib_samples: int = 128,
    seq_len: int = 2048,
    save_dir: str | None = None,
    tokenizer: Any | None = None,
    **_: object,
) -> nn.Module:
    """Quantize a causal LM with GPTQ.

    Parameters
    ----------
    model:
        Either an in-memory ``nn.Module`` (HF causal LM) or a model id / path
        understood by ``AutoGPTQForCausalLM.from_pretrained``.
    bits:
        2, 3, 4 or 8. 4 is the sweet spot for most LLMs.
    group_size:
        Number of weights per quantization group. -1 disables grouping.
    desc_act:
        Activation-order quantization. Slower, marginally better.
    calib_dataset:
        Built-in name (``"wikitext"``, ``"c4"``, ``"ptb"``) or a list of strings.
    calib_samples:
        Number of calibration sequences. 128 is the de-facto standard.
    """
    auto_gptq = _require("auto_gptq")
    transformers = _require("transformers")
    BaseQuantizeConfig = auto_gptq.BaseQuantizeConfig
    AutoGPTQForCausalLM = auto_gptq.AutoGPTQForCausalLM

    qconfig = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=desc_act,
        sym=sym,
    )

    if isinstance(model, str):
        gptq_model = AutoGPTQForCausalLM.from_pretrained(model, qconfig)
        model_id = model
    else:
        # Round-trip via from_pretrained: auto-gptq expects its own wrapper.
        raise NotImplementedError(
            "GPTQ currently requires a model id or path; pass `model='org/model'`."
        )

    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_fast=True)

    examples = _build_calibration(tokenizer, calib_dataset, calib_samples, seq_len)
    gptq_model.quantize(examples)

    if save_dir is not None:
        gptq_model.save_quantized(save_dir, use_safetensors=True)

    return gptq_model


def _require(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            f"{name} is required for GPTQ. Install with `pip install turboquant[gptq]`."
        ) from e


def _build_calibration(
    tokenizer: Any,
    dataset: str | list[str],
    n_samples: int,
    seq_len: int,
) -> list[dict]:
    """Tokenize a small calibration set into the format auto-gptq expects."""
    import random

    if isinstance(dataset, list):
        texts = dataset
    else:
        datasets = _require("datasets")
        if dataset == "wikitext":
            ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = [t for t in ds["text"] if t.strip()]
        elif dataset == "c4":
            ds = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = [next(iter(ds))["text"] for _ in range(n_samples * 4)]
        else:
            raise ValueError(f"Unknown calibration dataset '{dataset}'")

    random.shuffle(texts)
    examples: list[dict] = []
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
        if enc.input_ids.shape[1] < seq_len // 2:
            continue
        examples.append({"input_ids": enc.input_ids, "attention_mask": enc.attention_mask})
        if len(examples) >= n_samples:
            break
    return examples
