# Reference benchmark results

These JSON + PNG files are the output of `benchmarks/scripts/sweep_cpu.py` run
on a development workstation. They are committed so the figures in the project
README are reproducible and auditable.

| File | Model | Hardware | Methods |
|---|---|---|---|
| `smollm2_135m.json` / `.png` | `HuggingFaceTB/SmolLM2-135M` | Windows 11 / CPU / torch 2.12 | fp32, fp16, bf16, int8-dynamic |
| `gpt2.json` / `.png` | `gpt2` | Windows 11 / CPU / torch 2.12 | fp32, fp16, bf16, int8-dynamic |

To reproduce locally:

```bash
python benchmarks/scripts/sweep_cpu.py \
    --model-id HuggingFaceTB/SmolLM2-135M \
    --methods fp32,fp16,bf16,int8-dynamic \
    --out benchmarks/results/smollm2_135m.json \
    --plot benchmarks/results/smollm2_135m.png
```

To add results for your own hardware, drop a JSON + PNG pair here and
allowlist them in `.gitignore`.
