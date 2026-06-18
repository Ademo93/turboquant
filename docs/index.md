# TurboQuant

> Model quantization & optimization toolkit for edge and resource-constrained deployment.
> INT4 · INT8 · FP16 · GPTQ · AWQ · BitsAndBytes · Structured pruning · ONNX & TensorRT.

## Why TurboQuant?

Modern open-source models are powerful but expensive to serve. A 7B-parameter
LLM in FP16 demands ~14 GB of VRAM; a vision transformer that fits comfortably
on a workstation may blow up on a Jetson Orin or a phone. **TurboQuant gives
you a single, consistent interface to compress, quantize, prune, export, and
benchmark models** — so you can ship them on the hardware you actually have.

It is built around three principles:

1. **One API, many backends.** Wrap `bitsandbytes`, `auto-gptq`, `autoawq`,
   native PyTorch quantization, and ONNX/TensorRT export behind a uniform
   `quantize(model, method=...)` interface.
2. **Reproducible benchmarks.** Latency, peak memory, model size, and task
   accuracy (perplexity, top-k, etc.) are first-class citizens — every example
   ships with a comparable benchmark.
3. **No magic.** Each technique is a small, readable module so it doubles as a
   reference for *how* the method works.

## Install

```bash
pip install turboquant                       # core
pip install "turboquant[onnx]"               # + ONNX export
pip install "turboquant[gptq,awq,bnb,eval]"  # full LLM compression stack
pip install "turboquant[all]"                # everything
```

## 60-second tour

```python
from turboquant import quantize, benchmark
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

qmodel = quantize(base, method="int8-dynamic")

report = benchmark.compare(
    baseline=base,
    candidate=qmodel,
    tokenizer=tok,
    prompts=["Explain quantization in one sentence."],
    metrics=("latency", "memory", "size"),
    names=("fp32", "int8-dynamic"),
)
print(report.as_table())
```

```text
+------+---------+------------+------------+--------------+
| name | size_mb | latency_ms | throughput | peak_gpu_mb  |
+------+---------+------------+------------+--------------+
| fp32 | ...     | ...        | ...        | ...          |
| int8 | ...     | ...        | ...        | ...          |
+------+---------+------------+------------+--------------+
```

## CLI

```bash
tq quantize Qwen/Qwen2.5-0.5B --method bnb-nf4 --out outputs/qwen-nf4
tq prune microsoft/resnet-50 --strategy l1-channel --sparsity 0.3
tq export outputs/resnet50-pruned --format onnx --quant int8-dynamic
tq bench Qwen/Qwen2.5-0.5B --methods fp16,int8-dynamic --plot
tq methods
```

See **[Quantization](quantization.md)** for the algorithm reference,
**[Pruning](pruning.md)** for sparsity recipes, and
**[Benchmarks](benchmarks.md)** for the measurement methodology.
