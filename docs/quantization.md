# Quantization

TurboQuant exposes every backend through a single dispatcher:

```python
from turboquant import quantize
qmodel = quantize(model, method="bnb-nf4")
```

The `method` argument selects one of the registered backends. They all share
the same signature — `(model, **config) -> nn.Module` — so swapping methods is
a one-line change.

## Methods

### `fp16` / `bf16`

Cast every floating-point parameter and buffer to the target dtype. No
calibration, no accuracy loss to speak of, ~2× smaller weights, and faster
matmul on any Tensor-Core GPU. Always include this as your baseline.

```python
qmodel = quantize(model, method="fp16")
```

### `int8-dynamic`

PyTorch native dynamic INT8: weights are stored INT8 and dequantized on the
fly, activations are quantized at runtime. Targets `nn.Linear`, `nn.LSTM`,
`nn.GRU` — perfect for transformer encoders served on CPU.

```python
qmodel = quantize(model, method="int8-dynamic")
```

### `int8-static`

Static PTQ with calibration. You provide a small `calib_loader` of representative
batches and TurboQuant fits per-tensor or per-channel scale + zero-point values.

```python
qmodel = quantize(
    model,
    method="int8-static",
    calib_loader=my_loader,
    observer="histogram",
    per_channel=True,
)
```

### `bnb-int8` / `bnb-nf4` / `bnb-fp4`

BitsAndBytes weight-only quantization. Swaps `nn.Linear` for `Linear8bitLt` or
`Linear4bit`. The 4-bit variants store weights in 4-bit blocks with a small
secondary quantization of the block scales (`double_quant=True`), reaching
~3.5 bits per weight effective.

```python
qmodel = quantize(model, method="bnb-nf4", compute_dtype="float16", double_quant=True)
```

### `gptq`

GPTQ: solve a per-layer least-squares problem column-by-column using approximate
Hessian information. Wraps `auto-gptq`. Pass a HuggingFace model id; the wrapper
loads, calibrates, and saves a packed checkpoint.

```python
quantize(
    "meta-llama/Llama-3.2-1B",
    method="gptq",
    bits=4,
    group_size=128,
    calib_dataset="wikitext",
    save_dir="outputs/llama-gptq",
)
```

### `awq`

Activation-aware Weight Quantization. Wraps `autoawq`. Same call shape as GPTQ.

```python
quantize(
    "meta-llama/Llama-3.2-1B",
    method="awq",
    bits=4,
    save_dir="outputs/llama-awq",
)
```

## How to pick a method

| Constraint | Pick |
|---|---|
| CPU inference, transformer encoder | `int8-dynamic` |
| Edge CPU/NPU, CNN | `int8-static` |
| Single consumer GPU, LLM, want best ppl | `gptq` or `awq` |
| Training (LoRA / QLoRA) | `bnb-nf4` |
| Lossless-ish baseline | `fp16` or `bf16` |
| 2× speedup on Ampere/Hopper tensor cores | 2:4 N:M sparsity + INT8 |
