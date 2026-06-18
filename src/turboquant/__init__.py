"""TurboQuant — model quantization & optimization toolkit.

Public entry points:
    quantize(model, method=...)   one-shot quantization dispatch
    prune(model, strategy=...)    one-shot pruning dispatch
    benchmark                     latency / memory / accuracy reports
    export                        ONNX / TensorRT export helpers
"""

from importlib.metadata import PackageNotFoundError, version

from turboquant import benchmark, export, pruning, quantization
from turboquant.pruning import prune
from turboquant.quantization import quantize

try:
    __version__ = version("turboquant")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

__all__ = [
    "__version__",
    "benchmark",
    "export",
    "prune",
    "pruning",
    "quantization",
    "quantize",
]
