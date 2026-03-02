"""Unit tests for evals.render_results_table."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.render_results_table import render_table


def test_render_table_sorts_by_run_composite_desc(tmp_path: Path) -> None:
    """Table rows are sorted by run_composite descending (best first)."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"models": {'
        '"a": {"run_composite": 90.0, "p50_latency_ms": 1000}, '
        '"b": {"run_composite": 95.0, "p50_latency_ms": 2000}'
        "}}",
        encoding="utf-8",
    )
    table = render_table(summary)
    lines = table.strip().split("\n")
    # Header + alignment + 2 data rows
    assert len(lines) >= 4
    assert "| b |" in lines[2]  # 95.0 first
    assert "| a |" in lines[3]  # 90.0 second


def test_render_table_recommended_first(tmp_path: Path) -> None:
    """With --recommended, that model is the first data row."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"models": {'
        '"gpt-4o": {"run_composite": 96.0, "p50_latency_ms": 5000}, '
        '"gpt-4o-mini": {"run_composite": 93.0, "p50_latency_ms": 6000}'
        "}}",
        encoding="utf-8",
    )
    table = render_table(summary, recommended_model="gpt-4o-mini")
    lines = table.strip().split("\n")
    assert "| gpt-4o-mini |" in lines[2]
    assert "| gpt-4o |" in lines[3]


def test_render_table_missing_tokens_shows_em_dash(tmp_path: Path) -> None:
    """When avg_input_tokens/avg_output_tokens are missing, cell shows —."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"models": {"m": {"run_composite": 100.0, "p50_latency_ms": 1000}}}',
        encoding="utf-8",
    )
    table = render_table(summary)
    assert "—" in table
    # Table has Model | Run composite | Guardrails % | Safety % | Meal % | Avg in | Avg out | P50
    assert "| m | 100.0 | — | — | — | — | — | 1000 |" in table


def test_render_table_with_tokens(tmp_path: Path) -> None:
    """When token fields exist, they appear in the table."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"models": {"m": {"run_composite": 95.0, "avg_input_tokens": 500, '
        '"avg_output_tokens": 100, "p50_latency_ms": 2000}}}',
        encoding="utf-8",
    )
    table = render_table(summary)
    assert "| m | 95.0 | — | — | — | 500 | 100 | 2000 |" in table


def test_render_table_includes_agent_metrics(tmp_path: Path) -> None:
    """Table includes Guardrails %, Safety %, Meal % columns from summary."""
    summary = tmp_path / "summary.json"
    summary.write_text(
        '{"models": {"m": {"run_composite": 94.0, "guardrails_pct": 100.0, '
        '"safety_pct": 88.0, "meal_pct": 96.0, "p50_latency_ms": 5000}}}',
        encoding="utf-8",
    )
    table = render_table(summary)
    assert "Guardrails %" in table
    assert "Safety %" in table
    assert "Meal %" in table
    assert "| m | 94.0 | 100.0 | 88.0 | 96.0 | — | — | 5000 |" in table


def test_render_table_empty_models(tmp_path: Path) -> None:
    """Empty models dict yields a short message."""
    summary = tmp_path / "summary.json"
    summary.write_text('{"models": {}}', encoding="utf-8")
    table = render_table(summary)
    assert "No model data" in table
