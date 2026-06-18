"""Lightweight activation observers used by the static INT8 path.

PyTorch provides production-grade observers under ``torch.ao.quantization``;
the small reimplementation here exists to make the underlying math obvious for
readers and tests. They are functionally equivalent for common cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class QParams:
    """Result of fitting an observer: scale and zero-point."""

    scale: torch.Tensor
    zero_point: torch.Tensor
    qmin: int
    qmax: int

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.round(x / self.scale + self.zero_point)
        return q.clamp(self.qmin, self.qmax).to(torch.int8)

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return (q.float() - self.zero_point) * self.scale


class MinMaxObserver:
    """Tracks min/max of seen tensors and computes affine INT8 params."""

    def __init__(self, *, symmetric: bool = True, qmin: int = -128, qmax: int = 127) -> None:
        self.symmetric = symmetric
        self.qmin = qmin
        self.qmax = qmax
        self.min_val = torch.tensor(float("inf"))
        self.max_val = torch.tensor(float("-inf"))

    def observe(self, x: torch.Tensor) -> None:
        self.min_val = torch.minimum(self.min_val, x.min().detach())
        self.max_val = torch.maximum(self.max_val, x.max().detach())

    def compute(self) -> QParams:
        if self.symmetric:
            abs_max = torch.maximum(self.min_val.abs(), self.max_val.abs())
            scale = abs_max / max(abs(self.qmin), self.qmax)
            zero_point = torch.zeros_like(scale)
        else:
            scale = (self.max_val - self.min_val) / (self.qmax - self.qmin)
            zero_point = torch.round(self.qmin - self.min_val / scale)
        scale = torch.clamp(scale, min=1e-8)
        return QParams(scale=scale, zero_point=zero_point, qmin=self.qmin, qmax=self.qmax)


class PercentileObserver(MinMaxObserver):
    """Clip to the [p, 100-p] percentile to ignore outliers."""

    def __init__(self, *, percentile: float = 99.9, **kw: object) -> None:
        super().__init__(**kw)  # type: ignore[arg-type]
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be in (0, 100)")
        self.percentile = percentile
        self._cache: list[torch.Tensor] = []

    def observe(self, x: torch.Tensor) -> None:
        self._cache.append(x.detach().flatten())

    def compute(self) -> QParams:
        all_vals = torch.cat(self._cache)
        lo = torch.quantile(all_vals, (100 - self.percentile) / 100)
        hi = torch.quantile(all_vals, self.percentile / 100)
        self.min_val = lo
        self.max_val = hi
        return super().compute()


__all__ = ["MinMaxObserver", "PercentileObserver", "QParams"]
