"""Export PyTorch modules to ONNX with optional graph slimming and INT8 quantization."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch
from torch import nn


def export_onnx(
    model: nn.Module,
    sample_input: torch.Tensor | tuple[torch.Tensor, ...],
    out_path: str | Path,
    *,
    opset: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    input_names: tuple[str, ...] = ("input",),
    output_names: tuple[str, ...] = ("output",),
    slim: bool = True,
) -> Path:
    """Export ``model`` to ONNX and (optionally) optimize the graph with ``onnxslim``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval()

    if isinstance(sample_input, torch.Tensor):
        sample_input = (sample_input,)

    torch.onnx.export(
        model,
        sample_input,
        out_path.as_posix(),
        opset_version=opset,
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    if slim:
        try:
            onnxslim = importlib.import_module("onnxslim")
            onnxslim.slim(out_path.as_posix(), out_path.as_posix())
        except ImportError:
            pass  # optional

    return out_path


def quantize_onnx_dynamic(
    onnx_path: str | Path,
    out_path: str | Path,
    *,
    weight_type: str = "QInt8",
) -> Path:
    """Apply ONNX Runtime dynamic quantization to weights only."""
    ort_q = _require_ort_quantization()
    onnx_path = Path(onnx_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    weight_enum = getattr(ort_q.QuantType, weight_type)
    ort_q.quantize_dynamic(onnx_path.as_posix(), out_path.as_posix(), weight_type=weight_enum)
    return out_path


def _require_ort_quantization() -> Any:
    try:
        from onnxruntime import quantization as ort_q

        return ort_q
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "onnxruntime is required. Install with `pip install turboquant[onnx]`."
        ) from e
