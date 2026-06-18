# Benchmarks

Every quantization comparison should report at least: **size, latency, memory,
and a task metric**. TurboQuant ships helpers for all four.

```python
from turboquant.benchmark import compare

report = compare(
    baseline=fp16_model,
    candidate=int4_model,
    tokenizer=tok,
    prompts=["Explain quantization."],
    metrics=("latency", "memory", "size", "perplexity"),
    iters=50,
)
print(report.as_table())
report.save("benchmarks/results/llama-1b.json")
```

## Latency

`measure_latency(fn, warmup=10, iters=50)` warms the kernel up, then times
`iters` runs. On CUDA it uses cuda events with synchronization; on CPU it
falls back to `time.perf_counter`. Reports mean, median, p95, p99.

> Always discard at least 5 warmup runs — JIT compilation, kernel autotuning,
> and GPU clock boosting all happen on the first calls.

## Memory

`measure_memory()` is a context manager. On CUDA it resets
`cuda.max_memory_allocated()` and reads it at exit; on CPU it diffs `psutil`
RSS. Use it around the call that exercises the model, not around model
loading — you want peak inference-time memory.

## Model size

`model_size_bytes(model)` serializes the state dict to a temp file and reads
its size. This counts the *real* serialized bytes, including bitsandbytes
packed quantization state — which `sum(p.numel() * p.element_size())` would
miss.

## Perplexity

`perplexity(model, tokenizer, texts, max_length=2048, stride=1024)` implements
the sliding-window recipe from the HF docs. Lower is better.

## Reproducibility

- Pin the seed: `from turboquant.utils import seed_everything; seed_everything(0)`
- Pin the GPU clock: `nvidia-smi --lock-gpu-clocks=1500,1500`
- Report median + p95, not just mean
- Always include an FP16/BF16 baseline on the same hardware
