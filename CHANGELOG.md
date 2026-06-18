# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and adheres
to [Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.1.0] — 2026-06-18

### Added
- Unified `quantize(model, method=...)` dispatcher with FP16/BF16, INT8 dynamic
  & static, BitsAndBytes (INT8 / NF4 / FP4), GPTQ, AWQ backends.
- Unified `prune(model, strategy=...)` dispatcher with magnitude, L1/L2
  structured channel pruning, and 2:4 N:M sparsity.
- ONNX export with optional `onnxslim` graph optimization and ORT dynamic INT8
  weight quantization.
- TensorRT engine builder with FP16 / INT8 calibration support.
- Benchmark helpers: latency (CUDA events + p95/p99), peak GPU/CPU memory,
  serialized model size, sliding-window perplexity, top-k accuracy.
- Typer CLI with `quantize`, `prune`, `export`, `bench`, `methods` subcommands.
- pytest suite covering quantization, pruning, benchmark and CLI smoke paths.
- GitHub Actions CI (lint + tests on Python 3.10–3.12 + wheel build).
