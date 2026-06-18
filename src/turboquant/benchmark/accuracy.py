"""Task metrics — perplexity for LMs, top-k for classifiers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


@torch.no_grad()
def perplexity(
    model: nn.Module,
    tokenizer: Any,
    texts: Iterable[str],
    *,
    max_length: int = 2048,
    stride: int = 1024,
    device: str | None = None,
) -> float:
    """Sliding-window perplexity, à la the HuggingFace recipe.

    Uses overlapping windows of ``max_length`` tokens with ``stride`` step.
    Lower is better.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    full = tokenizer("\n\n".join(texts), return_tensors="pt").input_ids.to(device)
    seq_len = full.size(1)

    nlls: list[torch.Tensor] = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        ids = full[:, begin:end]
        targets = ids.clone()
        targets[:, :-trg_len] = -100
        out = model(ids, labels=targets)
        # HF returns mean over the trg_len-1 predicted tokens.
        nlls.append(out.loss.detach() * trg_len)
        prev_end = end
        if end == seq_len:
            break

    return math.exp(torch.stack(nlls).sum().item() / seq_len)


@torch.no_grad()
def top_k_accuracy(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    k: int = 1,
    device: str | None = None,
) -> float:
    """Standard top-k classification accuracy."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    correct = 0
    total = 0
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        if hasattr(logits, "logits"):
            logits = logits.logits
        _, pred = logits.topk(k, dim=-1)
        correct += (pred == y.unsqueeze(-1)).any(dim=-1).sum().item()
        total += y.numel()
    return correct / total if total else 0.0
