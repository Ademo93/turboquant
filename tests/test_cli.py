"""Smoke tests for the Typer CLI — verifies wiring without downloading models."""

from __future__ import annotations

from typer.testing import CliRunner

from turboquant.cli import app

runner = CliRunner()


def test_methods_subcommand_lists_known_quantizers() -> None:
    result = runner.invoke(app, ["methods"])
    assert result.exit_code == 0
    for expected in ("bnb-nf4", "gptq", "awq", "int8-dynamic", "l1-channel", "magnitude"):
        assert expected in result.stdout


def test_help_renders() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "quantize" in result.stdout
    assert "prune" in result.stdout
    assert "export" in result.stdout
    assert "bench" in result.stdout
