"""Quantize a small Llama-class model to INT4 with bitsandbytes and benchmark it.

Run:
    python examples/quantize_llm_int4.py --model-id meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from turboquant import benchmark, quantize
from turboquant.models import load_causal_lm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--method", default="bnb-nf4", choices=["bnb-int8", "bnb-nf4", "bnb-fp4"])
    parser.add_argument("--out", type=Path, default=Path("outputs/llm_int4"))
    parser.add_argument("--prompt", default="In one sentence, what is quantization?")
    args = parser.parse_args()

    base, tok = load_causal_lm(args.model_id)
    cand, _ = load_causal_lm(args.model_id)
    cand = quantize(cand, method=args.method)

    report = benchmark.compare(
        baseline=base,
        candidate=cand,
        tokenizer=tok,
        prompts=[args.prompt],
        metrics=("latency", "memory", "size"),
        names=("fp16", args.method),
    )
    print(report.as_table())

    args.out.mkdir(parents=True, exist_ok=True)
    report.save(args.out / "report.json")
    print(f"\nReport saved to {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
