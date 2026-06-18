"""GPTQ 4-bit weight-only quantization for a causal LM.

Requires:
    pip install "turboquant[gptq]"

Run:
    python examples/quantize_llm_gptq.py --model-id meta-llama/Llama-3.2-1B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from turboquant.quantization import gptq_int


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("outputs/llm_gptq"))
    args = parser.parse_args()

    print(f"GPTQ {args.bits}-bit quantization of {args.model_id} ...")
    gptq_int.quantize_gptq(
        args.model_id,
        bits=args.bits,
        group_size=args.group_size,
        calib_samples=args.calib_samples,
        save_dir=args.out.as_posix(),
    )
    print(f"✓ Quantized model saved to {args.out}")


if __name__ == "__main__":
    main()
