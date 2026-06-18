"""Benchmark FP16 baseline vs several quantization methods, save JSON + plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from turboquant import benchmark, quantize
from turboquant.models import load_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--methods", default="bnb-int8,bnb-nf4")
    parser.add_argument("--prompt", default="Briefly: what is INT8 quantization?")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    base, tok = load_causal_lm(args.model_id)
    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for method in methods:
        print(f"→ {method}")
        cand, _ = load_causal_lm(args.model_id)
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
        if not runs:
            runs.append(rep.runs[0])  # baseline once
        runs.append(rep.runs[1])

    out_json = args.out / f"{args.model_id.replace('/', '_')}.json"
    out_json.write_text(json.dumps([r.__dict__ for r in runs], indent=2))
    print(f"✓ Saved {out_json}")


if __name__ == "__main__":
    main()
