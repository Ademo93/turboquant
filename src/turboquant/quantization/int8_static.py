"""INT8 static post-training quantization.

Static PTQ records activation statistics on a small calibration set, then bakes
fixed scale/zero-point values into the model. It is the standard recipe for
CNNs targeting CPU or NPU runtimes.

Workflow
--------
1. ``prepare(model)`` inserts ``QuantStub`` / observers
2. The user runs a few hundred calibration batches through the prepared model
3. ``convert(model)`` swaps observed modules for quantized kernels
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


def quantize_int8_static(
    model: nn.Module,
    *,
    calib_loader: Iterable[Any] | None = None,
    backend: str = "fbgemm",
    observer: str = "minmax",
    per_channel: bool = True,
    **_: object,
) -> nn.Module:
    """Static INT8 PTQ.

    Parameters
    ----------
    calib_loader:
        Iterable that yields model inputs (tensor or tuple). 100-512 batches is
        typical. If ``None``, the function returns the *prepared* (observed)
        model so the caller can run calibration externally.
    backend:
        ``"fbgemm"`` for x86 CPU, ``"qnnpack"`` for ARM.
    observer:
        ``"minmax"`` (default), ``"histogram"`` (entropy-style), or
        ``"percentile"``.
    """
    torch.backends.quantized.engine = backend
    model = model.cpu().eval()

    qconfig = _make_qconfig(observer=observer, per_channel=per_channel)
    model.qconfig = qconfig  # type: ignore[assignment]

    prepared = torch.ao.quantization.prepare(model, inplace=False)

    if calib_loader is None:
        return prepared

    with torch.no_grad():
        for batch in calib_loader:
            if isinstance(batch, (tuple, list)):
                prepared(*batch)
            elif isinstance(batch, dict):
                prepared(**batch)
            else:
                prepared(batch)

    return torch.ao.quantization.convert(prepared, inplace=False)


def _make_qconfig(observer: str, per_channel: bool) -> Any:
    from torch.ao.quantization import (
        HistogramObserver,
        MinMaxObserver,
        PerChannelMinMaxObserver,
        QConfig,
    )

    if observer == "histogram":
        act = HistogramObserver.with_args(reduce_range=True)
    else:
        act = MinMaxObserver.with_args(reduce_range=True)

    if per_channel:
        weight = PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8, qscheme=torch.per_channel_symmetric
        )
    else:
        weight = MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)

    return QConfig(activation=act, weight=weight)
