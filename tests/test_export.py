"""Smoke test for ONNX export — skips cleanly if onnx is not installed."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn


def test_export_onnx_writes_file(tmp_path: Path, tiny_mlp: nn.Module) -> None:
    pytest.importorskip("onnx")
    from turboquant.export import export_onnx

    sample = torch.randn(1, 32)
    out = export_onnx(tiny_mlp.eval(), sample, tmp_path / "tiny.onnx", slim=False)
    assert out.exists()
    assert out.stat().st_size > 0


def test_onnx_dynamic_int8(tmp_path: Path, tiny_mlp: nn.Module) -> None:
    pytest.importorskip("onnxruntime")
    from turboquant.export import export_onnx, quantize_onnx_dynamic

    sample = torch.randn(1, 32)
    onnx_path = export_onnx(tiny_mlp.eval(), sample, tmp_path / "tiny.onnx", slim=False)
    qpath = quantize_onnx_dynamic(onnx_path, tmp_path / "tiny.int8.onnx")
    assert qpath.exists()
    assert qpath.stat().st_size < onnx_path.stat().st_size
