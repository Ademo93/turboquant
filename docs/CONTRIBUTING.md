# Contributing to TurboQuant

Thanks for considering a contribution. This document captures the few
conventions that keep the project pleasant to work in.

## Setup

```bash
git clone https://github.com/Ademo93/turboquant
cd turboquant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,onnx,viz]"
pre-commit install
```

## Running checks

```bash
ruff check src tests
ruff format src tests
pytest -m "not slow and not gpu"
```

The full GPU/slow suite is opt-in:

```bash
pytest -m gpu
pytest -m slow
```

## Adding a new quantization method

1. Drop a module under `src/turboquant/quantization/your_method.py` with a single
   public function `quantize_your_method(model, **kw) -> nn.Module`.
2. Register it in `src/turboquant/quantization/__init__.py` (`_REGISTRY`).
3. Add it to the `Method` literal type.
4. Write a unit test in `tests/test_quantization.py` (use the `tiny_mlp` fixture).
5. Document it in `docs/quantization.md`.
6. If the backend depends on a heavy library, add a `[project.optional-dependencies]`
   entry in `pyproject.toml` and use `importlib.import_module` lazily.

## Coding style

- Keep modules short and readable. The goal is that each algorithm doubles as a
  reference for *how* the method works.
- Cite the original paper in the module docstring.
- Prefer explicit kwargs over `**kw` dicts in public APIs.
- No top-level imports of optional heavy dependencies (bitsandbytes,
  auto-gptq, tensorrt). Defer them inside functions.

## Pull request checklist

- [ ] Tests pass: `pytest -m "not slow and not gpu"`
- [ ] Lint passes: `ruff check src tests && ruff format --check src tests`
- [ ] New public APIs documented under `docs/`
- [ ] If adding a backend, it degrades cleanly when its optional dependency
      is missing (raises `ImportError` with install instructions)
