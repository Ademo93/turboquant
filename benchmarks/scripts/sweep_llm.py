"""Sweep a model across multiple quantization methods and emit a results CSV.

Run:
    python benchmarks/scripts/sweep_llm.py \
        --model-id meta-llama/Llama-3.2-1B \
        --methods fp16,bnb-int8,bnb-nf4,gptq,awq \
        --out benchmarks/results/llama32-1b.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from turboquant import benchmark, quantize
from turboquant.models import load_causal_lm
from turboquant.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--methods", default="fp16,bnb-int8,bnb-nf4")
    parser.add_argument("--prompt", default="Summarize quantization in 50 words.")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    base, tok = load_causal_lm(args.model_id)

    rows = []
    for method in methods:
        print(f"→ {method}")
        cand, _ = load_causal_lm(args.model_id)
        if method != "fp16":  # baseline already FP16/auto
            cand = quantize(cand, method=method)
        rep = benchmark.compare(
            base,
            cand,
            tokenizer=tok,
            prompts=[args.prompt],
            metrics=("latency", "memory", "size"),
            iters=args.iters,
            names=("fp16", method),
        )
        rows.append(rep.runs[1])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "size_mb", "latency_ms", "throughput", "peak_gpu_mb"])
        for r in rows:
            writer.writerow([r.name, r.size_mb, r.latency_ms, r.throughput, r.peak_gpu_mb])

    print(f"✓ Wrote {args.out}")


if __name__ == "__main__":
    main()
