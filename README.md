<h1 align="center">TurboQuant</h1>

<p align="center">
  <strong>Model quantization & optimization toolkit for edge and resource-constrained deployment.</strong><br>
  INT4 · INT8 · FP16 · GPTQ · AWQ · BitsandBytes · Structured pruning · ONNX & TensorRT export
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c"></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"></a>
  <a href="https://github.com/Ademo93/turboquant/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Ademo93/turboquant/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://Ademo93.github.io/turboquant/"><img alt="Docs" src="https://img.shields.io/badge/docs-mkdocs--material-blue"></a>
  <a href="#"><img alt="Status" src="https://img.shields.io/badge/status-beta-orange"></a>
</p>

---

## Why TurboQuant?

Modern open-source models are powerful but expensive to serve. Shipping a 7B-parameter LLM in FP16 demands ~14&nbsp;GB of VRAM; a vision transformer that fits comfortably on a workstation may blow up on a Jetson Orin or a phone. **TurboQuant gives you a single, consistent interface to compress, quantize, prune, export, and benchmark models** — so you can ship them on the hardware you actually have.

It is built around three principles:

1. **One API, many backends.** Wrap `bitsandbytes`, `auto-gptq`, `autoawq`, native PyTorch quantization, and ONNX/TensorRT export behind a uniform `quantize(model, method=...)` interface.
2. **Reproducible benchmarks.** Latency, peak memory, model size, and task accuracy (perplexity, classification top-1, etc.) are first-class citizens — every example ships with a comparable benchmark.
3. **No magic.** Each technique is implemented as a small, readable module so it doubles as a reference for how the methods work.

## Features

| Category | Techniques |
|---|---|
| **Weight quantization** | INT8 dynamic & static PTQ, FP16/BF16 casting, INT4 (bitsandbytes NF4 / FP4), GPTQ, AWQ |
| **Pruning** | Magnitude (unstructured), L1 structured (channel/filter), N:M sparsity helpers |
| **Export** | ONNX (with `onnxslim` graph optimization), TensorRT engine builder, ORT quantization |
| **Calibration** | Per-tensor & per-channel, MinMax / Entropy / Percentile observers |
| **Benchmark** | Latency (warmup + median + p95), peak GPU/CPU memory, throughput, model size, perplexity, top-k accuracy |
| **CLI** | `turboquant quantize`, `turboquant prune`, `turboquant export`, `turboquant bench` |

## Installation

```bash
# Core install
pip install turboquant

# With ONNX export
pip install "turboquant[onnx]"

# Full LLM compression stack (GPTQ + AWQ + bitsandbytes)
pip install "turboquant[gptq,awq,bnb,eval]"

# Everything
pip install "turboquant[all]"
```

> **Note** — `bitsandbytes`, `auto-gptq`, `autoawq` and `tensorrt` are heavy native dependencies. They are deliberately optional; TurboQuant degrades gracefully when they are missing.

## Quick start

### Python API

```python
from turboquant import quantize, benchmark
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

# One-line INT4 weight-only quantization via bitsandbytes
qmodel = quantize(model, method="bnb-nf4")

# Benchmark side-by-side
report = benchmark.compare(
    baseline=model,
    candidate=qmodel,
    tokenizer=tok,
    prompts=["Explain quantization in one sentence."],
    metrics=["latency", "memory", "size", "perplexity"],
)
print(report.as_table())
```

### CLI

```bash
# Quantize a HuggingFace model to INT4 with GPTQ + W4A16
tq quantize meta-llama/Llama-3.2-1B \
    --method gptq \
    --bits 4 \
    --group-size 128 \
    --calib-dataset wikitext \
    --out ./outputs/llama-3.2-1b-gptq

# Structured prune a vision model and re-evaluate
tq prune microsoft/resnet-50 \
    --strategy l1-channel \
    --sparsity 0.30 \
    --eval imagenet-val \
    --out ./outputs/resnet50-pruned

# Export to ONNX with INT8 dynamic quantization
tq export ./outputs/resnet50-pruned \
    --format onnx \
    --quant int8-dynamic \
    --opset 17

# Benchmark FP16 vs INT8 vs INT4 on a model
tq bench meta-llama/Llama-3.2-1B --methods fp16,int8-dynamic,bnb-nf4 \
    --report ./benchmarks/results/llama32-1b.json
```

## Supported methods at a glance

| Method | Bits | Backend | Calibration | Typical use case |
|---|---|---|---|---|
| `fp16` / `bf16` | 16 | PyTorch | none | Fast, lossless-ish baseline |
| `int8-dynamic` | 8 | PyTorch | none | CPU inference, transformers |
| `int8-static` | 8 | PyTorch | required | CNNs, edge CPUs |
| `bnb-int8` | 8 | bitsandbytes | none | LLM training & serving on GPU |
| `bnb-nf4` / `bnb-fp4` | 4 | bitsandbytes | none | LLM inference, QLoRA |
| `gptq` | 2–8 | auto-gptq | required | LLM weight-only, best accuracy/bit |
| `awq` | 4 | autoawq | required | LLM weight-only, fast inference |

## Reference benchmarks

> Numbers below are illustrative — re-run `tq bench` on your hardware for ground truth.

### Llama-3.2-1B on a single RTX 4070

| Method | Size | VRAM | Latency (tok/s) | Perplexity (wikitext) |
|---|---:|---:|---:|---:|
| FP16 baseline | 2.5 GB | 3.1 GB | 92 | 9.18 |
| INT8 (bnb) | 1.3 GB | 1.8 GB | 88 | 9.21 |
| GPTQ INT4 g128 | 0.9 GB | 1.4 GB | 121 | 9.37 |
| AWQ INT4 | 0.9 GB | 1.4 GB | 134 | 9.32 |
| NF4 (bnb) | 0.8 GB | 1.3 GB | 76 | 9.41 |

### ResNet-50 on ImageNet-val

| Method | Size | Top-1 | CPU latency (ms) |
|---|---:|---:|---:|
| FP32 baseline | 98 MB | 76.13% | 38.4 |
| FP16 | 49 MB | 76.10% | — |
| INT8 PTQ static | 25 MB | 75.62% | 11.7 |
| L1 channel-prune 30% + INT8 | 17 MB | 74.81% | 8.9 |

Run on your machine:

```bash
tq bench meta-llama/Llama-3.2-1B --methods fp16,gptq,awq,bnb-nf4 --plot
```

## Architecture

```
turboquant/
├── quantization/          # Algorithms: int8, fp16, gptq, awq, bnb, observers
├── pruning/               # Magnitude + structured (L1, L2, taylor) + N:M
├── export/                # ONNX, TensorRT, ORT quantization
├── benchmark/             # Latency, memory, perplexity, classification, plot
├── calibration/           # Datasets, dataloaders, observer fitting
├── models/                # Convenience loaders + registry
└── cli.py                 # Typer-based CLI
```

Each algorithm lives in a single, readable file with a `quantize_*` / `prune_*` function and a short docstring referencing the original paper.

## Roadmap

- [x] INT8 dynamic & static PTQ (PyTorch native)
- [x] FP16/BF16 casting
- [x] BitsAndBytes INT8 / NF4 / FP4 wrappers
- [x] GPTQ & AWQ integration
- [x] L1 structured & magnitude pruning
- [x] ONNX export with `onnxslim`
- [x] Latency / memory / perplexity benchmarks
- [ ] TensorRT INT8 calibration cache
- [ ] SmoothQuant W8A8
- [ ] HQQ (Half-Quadratic Quantization)
- [ ] Distillation-aware quantization
- [ ] Mobile export (CoreML / TFLite)
- [ ] Web dashboard for benchmark comparison

## Citing & related work

TurboQuant stands on the shoulders of giants. If you use it in research, please also cite the underlying algorithms:

- **GPTQ** — Frantar et al., 2023 (arXiv:2210.17323)
- **AWQ** — Lin et al., 2023 (arXiv:2306.00978)
- **LLM.int8()** / **QLoRA** — Dettmers et al., 2022 / 2023 (arXiv:2208.07339, 2305.14314)
- **SmoothQuant** — Xiao et al., 2022 (arXiv:2211.10438)

## Contributing

Contributions are very welcome — see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md). Good first issues are tagged on the issue tracker.

```bash
git clone https://github.com/Ademo93/turboquant
cd turboquant
pip install -e ".[dev,all]"
pre-commit install
pytest
```

## License

[MIT](LICENSE) — do whatever you like, just keep the copyright notice.
