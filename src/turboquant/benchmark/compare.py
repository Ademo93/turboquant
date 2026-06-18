"""Side-by-side benchmark report builder."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from turboquant.benchmark.accuracy import perplexity
from turboquant.benchmark.latency import measure_latency
from turboquant.benchmark.memory import measure_memory, model_size_bytes


@dataclass
class RunMetrics:
    name: str
    size_mb: float | None = None
    latency_ms: float | None = None
    throughput: float | None = None
    peak_gpu_mb: float | None = None
    peak_cpu_mb: float | None = None
    perplexity: float | None = None
    accuracy: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchReport:
    runs: list[RunMetrics]
    device: str

    def as_table(self) -> str:
        cols = ["name", "size_mb", "latency_ms", "throughput", "peak_gpu_mb", "perplexity"]
        rows = [[str(getattr(r, c, "")) for c in cols] for r in self.runs]
        widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
        sep = "+".join("-" * (w + 2) for w in widths)
        header = "|".join(f" {c:<{w}} " for c, w in zip(cols, widths, strict=True))
        body = "\n".join("|".join(f" {r[i]:<{w}} " for i, w in enumerate(widths)) for r in rows)
        return f"{sep}\n{header}\n{sep}\n{body}\n{sep}"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"device": self.device, "runs": [asdict(r) for r in self.runs]}, indent=2)
        )


def compare(
    baseline: nn.Module,
    candidate: nn.Module,
    *,
    sample_input: torch.Tensor | None = None,
    tokenizer: Any | None = None,
    prompts: list[str] | None = None,
    metrics: tuple[str, ...] = ("latency", "memory", "size"),
    warmup: int = 5,
    iters: int = 20,
    device: str | None = None,
    names: tuple[str, str] = ("baseline", "candidate"),
) -> BenchReport:
    """Run the requested metrics for two models and return a comparable report."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    runs = []
    for name, model in zip(names, (baseline, candidate), strict=True):
        run = RunMetrics(name=name)

        if "size" in metrics:
            run.size_mb = round(model_size_bytes(model) / 1024**2, 3)

        if "latency" in metrics:
            model.to(device).eval()
            inp = sample_input.to(device) if sample_input is not None else None
            if inp is None and tokenizer is not None and prompts:
                ids = tokenizer(prompts[0], return_tensors="pt").input_ids.to(device)

                def fn(m=model, x=ids):
                    return m.generate(x, max_new_tokens=32, do_sample=False)
            elif inp is not None:

                def fn(m=model, x=inp):
                    return m(x)
            else:
                raise ValueError("Need either `sample_input` or `(tokenizer, prompts)`")

            with measure_memory(device=device) as mem:
                lat = measure_latency(fn, warmup=warmup, iters=iters, device=device)

            run.latency_ms = round(lat.median_ms, 3)
            run.throughput = round(lat.throughput(), 2)
            if "memory" in metrics:
                run.peak_gpu_mb = round(mem.peak_gpu / 1024**2, 2)
                run.peak_cpu_mb = round(mem.peak_cpu / 1024**2, 2)

        if "perplexity" in metrics and tokenizer is not None and prompts:
            run.perplexity = round(perplexity(model, tokenizer, prompts), 3)

        runs.append(run)

    return BenchReport(runs=runs, device=device)
