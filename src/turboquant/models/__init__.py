"""Convenience model loaders.

These are very thin wrappers around HuggingFace ``from_pretrained`` that
default to sensible dtypes and device maps and make examples shorter.
"""

from __future__ import annotations

from typing import Any

from turboquant.utils import get_device


def load_causal_lm(
    model_id: str,
    *,
    dtype: str = "auto",
    device: str | None = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
):
    """Load a causal LM + tokenizer pair."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device or get_device(),
        trust_remote_code=trust_remote_code,
        **kwargs,
    )
    return model, tok


def load_image_classifier(
    model_id: str = "microsoft/resnet-50",
    *,
    device: str | None = None,
    **kwargs: Any,
):
    """Load an image classifier + processor pair."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id, **kwargs)
    model.to(device or get_device())
    return model, proc


__all__ = ["load_causal_lm", "load_image_classifier"]
