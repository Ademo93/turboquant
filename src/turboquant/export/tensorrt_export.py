"""TensorRT engine builder.

This is a thin wrapper around the TensorRT Python API. It supports FP16 and
INT8 with an explicit calibration cache. TensorRT is an optional extra and is
not installed by default — the function raises a clear error when missing.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def build_tensorrt_engine(
    onnx_path: str | Path,
    out_path: str | Path,
    *,
    precision: str = "fp16",
    workspace_gb: float = 4.0,
    calibrator: Any | None = None,
    max_batch_size: int = 1,
    profile_shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]
    | None = None,
) -> Path:
    """Build a TensorRT engine from an ONNX file.

    Parameters
    ----------
    precision:
        ``"fp32"``, ``"fp16"``, or ``"int8"``. INT8 requires ``calibrator``.
    profile_shapes:
        For dynamic-shape inputs: ``{input_name: (min, opt, max)}``.
    """
    trt = _require_trt()
    onnx_path = Path(onnx_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"Failed to parse ONNX: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if calibrator is None:
            raise ValueError("INT8 precision requires a calibrator")
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = calibrator
    elif precision != "fp32":
        raise ValueError(f"Unknown precision '{precision}'")

    if profile_shapes:
        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in profile_shapes.items():
            profile.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(profile)

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TensorRT failed to build the engine")
    out_path.write_bytes(engine_bytes)
    return out_path


def _require_trt() -> Any:
    try:
        return importlib.import_module("tensorrt")
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "TensorRT is not installed. Install NVIDIA's `tensorrt` Python package separately."
        ) from e


class NumpyCalibrator:
    """Minimal INT8 calibrator using an iterable of numpy arrays.

    Subclass and override :meth:`get_batch` for non-trivial datasets.
    """

    def __init__(self, batches: Iterable[Any], cache_file: str | Path = "calib.cache") -> None:
        trt = _require_trt()

        class _Inner(trt.IInt8EntropyCalibrator2):  # type: ignore[misc]
            def __init__(self_, iterator, cache_file):
                trt.IInt8EntropyCalibrator2.__init__(self_)
                self_._iter = iter(iterator)
                self_._cache_file = Path(cache_file)
                self_._device_buffer = None

            def get_batch_size(self_):
                return 1

            def get_batch(self_, _names):
                try:
                    batch = next(self_._iter)
                except StopIteration:
                    return None
                import pycuda.driver as cuda

                if self_._device_buffer is None:
                    self_._device_buffer = cuda.mem_alloc(batch.nbytes)
                cuda.memcpy_htod(self_._device_buffer, batch.tobytes())
                return [int(self_._device_buffer)]

            def read_calibration_cache(self_):
                if self_._cache_file.exists():
                    return self_._cache_file.read_bytes()
                return None

            def write_calibration_cache(self_, cache):
                self_._cache_file.write_bytes(cache)

        self._impl = _Inner(batches, cache_file)

    def __getattr__(self, item: str):
        return getattr(self._impl, item)
