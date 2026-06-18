"""Command-line interface — ``turboquant`` / ``tq``.

Subcommands:
    quantize    apply a quantization method to a HuggingFace model
    prune       apply a pruning strategy
    export      export to ONNX (optionally with INT8 quantization)
    bench       run latency / memory / accuracy comparisons
    methods     list available methods and strategies
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="turboquant",
    help="Model quantization & optimization toolkit.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def quantize(
    model_id: Annotated[str, typer.Argument(help="HuggingFace model id or local path")],
    method: Annotated[str, typer.Option(help="Quantization method")] = "bnb-nf4",
    bits: Annotated[int, typer.Option(help="Bit-width (where applicable)")] = 4,
    group_size: Annotated[int, typer.Option(help="GPTQ/AWQ group size")] = 128,
    calib_dataset: Annotated[str, typer.Option(help="Calibration dataset")] = "wikitext",
    calib_samples: Annotated[int, typer.Option()] = 128,
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")] = Path(
        "outputs/quantized"
    ),
) -> None:
    """Quantize a model and save the result."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turboquant import quantize as q
    from turboquant.benchmark import model_size_bytes

    console.print(
        Panel.fit(f"[bold cyan]turboquant quantize[/]  [dim]{model_id}[/]  →  [yellow]{method}[/]")
    )

    if method in {"gptq", "awq"}:
        qmodel = q(
            model_id,
            method=method,
            bits=bits,
            group_size=group_size,
            calib_dataset=calib_dataset,
            calib_samples=calib_samples,
            save_dir=out.as_posix(),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
        qmodel = q(model, method=method, bits=bits, group_size=group_size)
        out.mkdir(parents=True, exist_ok=True)
        qmodel.save_pretrained(out.as_posix())
        AutoTokenizer.from_pretrained(model_id).save_pretrained(out.as_posix())

    size_mb = model_size_bytes(qmodel) / 1024**2
    console.print(f"[green]✓[/] Saved to [bold]{out}[/]   ([yellow]{size_mb:.1f} MB[/])")


@app.command()
def prune(
    model_id: Annotated[str, typer.Argument(help="HuggingFace model id or local path")],
    strategy: Annotated[str, typer.Option(help="Pruning strategy")] = "l1-channel",
    sparsity: Annotated[float, typer.Option(help="Target sparsity in [0,1)")] = 0.3,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("outputs/pruned"),
) -> None:
    """Apply a pruning strategy and save the result."""
    from transformers import AutoModel, AutoTokenizer

    from turboquant import prune as p
    from turboquant.pruning import sparsity as measure

    console.print(
        Panel.fit(
            f"[bold cyan]turboquant prune[/]  [dim]{model_id}[/]  →  [yellow]{strategy} @ {sparsity:.0%}[/]"
        )
    )

    model = AutoModel.from_pretrained(model_id)
    pruned = p(model, strategy=strategy, sparsity=sparsity)
    out.mkdir(parents=True, exist_ok=True)
    pruned.save_pretrained(out.as_posix())
    with contextlib.suppress(Exception):
        AutoTokenizer.from_pretrained(model_id).save_pretrained(out.as_posix())

    console.print(
        f"[green]✓[/] Final sparsity: [yellow]{measure(pruned):.1%}[/]  →  [bold]{out}[/]"
    )


@app.command()
def export(
    model_id: Annotated[str, typer.Argument(help="Local path or HF id")],
    fmt: Annotated[str, typer.Option("--format", help="onnx|tensorrt")] = "onnx",
    quant: Annotated[str | None, typer.Option(help="onnx-int8-dynamic")] = None,
    opset: Annotated[int, typer.Option()] = 17,
    out: Annotated[Path, typer.Option("--out", "-o")] = Path("outputs/exported"),
) -> None:
    """Export to ONNX (optionally with INT8) or TensorRT."""
    from transformers import AutoModel, AutoTokenizer

    from turboquant.export import export_onnx, quantize_onnx_dynamic

    console.print(Panel.fit(f"[bold cyan]turboquant export[/]  →  [yellow]{fmt}[/]"))

    out.mkdir(parents=True, exist_ok=True)
    if fmt == "onnx":
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        sample = tok("hello world", return_tensors="pt").input_ids
        onnx_path = export_onnx(model, sample, out / "model.onnx", opset=opset)
        if quant == "int8-dynamic":
            qpath = quantize_onnx_dynamic(onnx_path, out / "model.int8.onnx")
            console.print(f"[green]✓[/] Quantized ONNX → [bold]{qpath}[/]")
        else:
            console.print(f"[green]✓[/] ONNX → [bold]{onnx_path}[/]")
    elif fmt == "tensorrt":
        console.print(
            "[yellow]TensorRT export expects an existing ONNX file. Use `tq export --format onnx` first.[/]"
        )
        raise typer.Exit(1)
    else:
        console.print(f"[red]Unknown format '{fmt}'[/]")
        raise typer.Exit(2)


@app.command()
def bench(
    model_id: Annotated[str, typer.Argument()],
    methods: Annotated[
        str, typer.Option(help="Comma-separated list of methods to compare against the baseline")
    ] = "fp16,int8-dynamic",
    prompt: Annotated[str, typer.Option()] = "Explain quantization in one sentence.",
    iters: Annotated[int, typer.Option()] = 20,
    report: Annotated[Path | None, typer.Option(help="Write JSON report here")] = None,
    plot: Annotated[bool, typer.Option(help="Save a matplotlib bar chart")] = False,
) -> None:
    """Benchmark a baseline model against one or more quantization methods."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from turboquant import quantize as q
    from turboquant.benchmark import compare

    console.print(Panel.fit(f"[bold cyan]turboquant bench[/]  [dim]{model_id}[/]"))

    method_list = [m.strip() for m in methods.split(",") if m.strip()]
    tok = AutoTokenizer.from_pretrained(model_id)
    base = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")

    all_runs = []
    for method in method_list:
        console.print(f"  → [yellow]{method}[/]")
        candidate = q(
            AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto"),
            method=method,
        )
        rep = compare(
            base,
            candidate,
            tokenizer=tok,
            prompts=[prompt],
            metrics=("latency", "memory", "size"),
            iters=iters,
            names=("fp16-baseline", method),
        )
        all_runs.extend(rep.runs[1:])  # avoid duplicating the baseline

    # Show baseline first, then candidates.
    base_rep = compare(
        base,
        base,
        tokenizer=tok,
        prompts=[prompt],
        metrics=("latency", "memory", "size"),
        iters=iters,
        names=("fp16-baseline", "fp16-baseline"),
    )
    table = Table(title="Benchmark results")
    for col in ("name", "size_mb", "latency_ms", "throughput", "peak_gpu_mb"):
        table.add_column(col)
    for r in [base_rep.runs[0], *all_runs]:
        table.add_row(
            r.name, str(r.size_mb), str(r.latency_ms), str(r.throughput), str(r.peak_gpu_mb)
        )
    console.print(table)

    if report is not None:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps([r.__dict__ for r in [base_rep.runs[0], *all_runs]], indent=2))
        console.print(f"[green]✓[/] Report saved to [bold]{report}[/]")

    if plot:
        _plot_bench([base_rep.runs[0], *all_runs], (report or Path("bench")).with_suffix(".png"))


def _plot_bench(runs, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]Install matplotlib (`pip install turboquant[viz]`) to plot.[/]")
        return
    names = [r.name for r in runs]
    latencies = [r.latency_ms or 0 for r in runs]
    sizes = [r.size_mb or 0 for r in runs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(names, latencies, color="steelblue")
    ax1.set_title("Median latency (ms)")
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(names, sizes, color="seagreen")
    ax2.set_title("Model size (MB)")
    ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    console.print(f"[green]✓[/] Plot saved to [bold]{out_path}[/]")


@app.command()
def methods() -> None:
    """List available quantization methods and pruning strategies."""
    from turboquant.pruning import list_strategies
    from turboquant.quantization import list_methods

    table = Table(title="Available methods")
    table.add_column("kind")
    table.add_column("name")
    for m in list_methods():
        table.add_row("quantize", m)
    for s in list_strategies():
        table.add_row("prune", s)
    console.print(table)


if __name__ == "__main__":
    app()
