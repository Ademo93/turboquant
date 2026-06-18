"""CPU-only sweep — works without CUDA or bitsandbytes.

Benchmarks FP32 (baseline) vs FP16, BF16, and INT8 dynamic on a small
non-gated causal LM. Produces JSON + a matplotlib bar chart.

Run:
    python benchmarks/scripts/sweep_cpu.py \
        --model-id HuggingFaceTB/SmolLM2-135M \
        --out benchmarks/results/smollm2_135m.json
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

try:
    import truststore  # Use OS cert store on Windows / corporate networks.
    truststore.inject_into_ssl()
except ImportError:
    pass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from turboquant import quantize
from turboquant.benchmark import measure_latency, model_size_bytes
from turboquant.utils import seed_everything


def load_fp32(model_id: str):
    return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--methods", default="fp32,fp16,bf16,int8-dynamic")
    parser.add_argument("--prompt", default="In one sentence, what is quantization?")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results/sweep_cpu.json"))
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed_everything(args.seed)
    print(f"Loading tokenizer + model: {args.model_id}")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    ids = tok(args.prompt, return_tensors="pt").input_ids
    runs: list[dict] = []

    for method in methods:
        print(f"\n-> {method}")
        t0 = time.perf_counter()
        model = load_fp32(args.model_id)

        if method == "fp32":
            cand = model
        elif method == "fp16":
            cand = copy.deepcopy(model).half()
            ids_run = ids
        elif method == "bf16":
            cand = copy.deepcopy(model).to(torch.bfloat16)
            ids_run = ids
        elif method == "int8-dynamic":
            cand = quantize(copy.deepcopy(model), method="int8-dynamic")
        else:
            cand = quantize(copy.deepcopy(model), method=method)

        ids_run = ids
        cand.eval()

        # Cast inputs only if we cast the model itself.
        with torch.no_grad():
            size_b = model_size_bytes(cand)

            def fwd(c=cand, x=ids_run):
                return c(x)

            lat = measure_latency(fwd, warmup=args.warmup, iters=args.iters, device="cpu")

            # Single greedy-ish generation for tokens/sec measurement.
            gen_t0 = time.perf_counter()
            out_ids = cand.generate(
                ids_run,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            gen_ms = (time.perf_counter() - gen_t0) * 1000
            new_tokens = out_ids.shape[-1] - ids_run.shape[-1]
            tok_per_s = new_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0

        elapsed = time.perf_counter() - t0
        row = {
            "method": method,
            "size_mb": round(size_b / 1024**2, 3),
            "fwd_latency_median_ms": round(lat.median_ms, 3),
            "fwd_latency_p95_ms": round(lat.p95_ms, 3),
            "gen_total_ms": round(gen_ms, 2),
            "gen_new_tokens": int(new_tokens),
            "gen_tok_per_s": round(tok_per_s, 2),
            "wall_s": round(elapsed, 2),
        }
        print(json.dumps(row, indent=2))
        runs.append(row)

        del cand, model
        import gc

        gc.collect()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "device": "cpu",
        "torch": torch.__version__,
        "runs": runs,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {args.out}")

    if args.plot is not None:
        _plot(runs, args.plot, args.model_id)


def _plot(runs: list[dict], out_path: Path, model_id: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return

    names = [r["method"] for r in runs]
    sizes = [r["size_mb"] for r in runs]
    latencies = [r["fwd_latency_median_ms"] for r in runs]
    tps = [r["gen_tok_per_s"] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].bar(names, sizes, color="seagreen")
    axes[0].set_ylabel("MB")
    axes[0].set_title("Serialized size")
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].bar(names, latencies, color="steelblue")
    axes[1].set_ylabel("ms")
    axes[1].set_title("Forward latency (median)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[2].bar(names, tps, color="darkorange")
    axes[2].set_ylabel("tokens/s")
    axes[2].set_title("Greedy generation throughput")
    axes[2].tick_params(axis="x", rotation=30)
    fig.suptitle(f"TurboQuant sweep — {model_id} — CPU")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
