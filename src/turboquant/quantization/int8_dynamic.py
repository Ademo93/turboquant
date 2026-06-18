"""INT8 dynamic post-training quantization (PyTorch native).

Dynamic quantization quantizes weights ahead of time and activations on the fly
during inference. It targets transformer- and RNN-style models on CPU, where
weight bandwidth is the bottleneck.

Reference: PyTorch quantization docs, https://pytorch.org/docs/stable/quantization.html
"""

from __future__ import annotations

import torch
from torch import nn


def quantize_int8_dynamic(
    model: nn.Module,
    *,
    target_layers: tuple[type[nn.Module], ...] = (nn.Linear, nn.LSTM, nn.GRU),
    dtype: torch.dtype = torch.qint8,
    **_: object,
) -> nn.Module:
    """Apply PyTorch's dynamic INT8 quantization to ``model``.

    Parameters
    ----------
    model:
        Module to quantize. The function moves it to CPU first because dynamic
        quantization currently runs only on CPU.
    target_layers:
        Layer types eligible for dynamic quantization. Linear/LSTM/GRU are the
        production-tested set; convolutions are not supported by this path.
    dtype:
        ``torch.qint8`` (default) or ``torch.quint8``.
    """
    model = model.cpu().eval()
    qmodel = torch.ao.quantization.quantize_dynamic(
        model,
        qconfig_spec=set(target_layers),
        dtype=dtype,
    )
    return qmodel
