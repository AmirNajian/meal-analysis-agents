"""Render eval metrics summary JSON as a Markdown results table.

Usage:
  uv run python -m evals.render_results_table [--summary FILE] [--output FILE] [--recommended MODEL]

Reads the summary JSON (default: data/results/eval_metrics_summary.json), sorts models
by run_composite descending (best first), and prints or writes a Markdown table.
Use --recommended to force a specific model as the first row.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


def _default_summary_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "results" / "eval_metrics_summary.json"


def _format_cell(value: str | int | float | None) -> str:
    """Format a table cell; use — for missing/None."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def render_table(summary_path: Path, recommended_model: str | None = None) -> str:
    """Build Markdown table from summary JSON. Sorted by run_composite desc; optional recommended first."""
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    models_data = data.get("models") or {}
    if not models_data:
        return "# Eval results\n\nNo model data in summary.\n"

    # Build rows: (model_name, run_composite, avg_input_tokens, avg_output_tokens, p50_latency_ms)
    rows = []
    for model_name, m in models_data.items():
        run_composite = m.get("run_composite")
        avg_in = m.get("avg_input_tokens")
        avg_out = m.get("avg_output_tokens")
        p50 = m.get("p50_latency_ms")
        rows.append((model_name, run_composite, avg_in, avg_out, p50))

    # Sort by run_composite descending (best first); put recommended first if set
    def sort_key(item: tuple) -> tuple:
        name, composite, *_ = item
        if recommended_model and name == recommended_model:
            return (0, -(composite or 0))  # recommended first, then by composite
        return (1, -(composite or 0))  # others by composite desc

    rows.sort(key=sort_key)

    # Header and alignment
    lines = [
        "| Model | Run composite | Avg input tokens | Avg output tokens | P50 latency (ms) |",
        "| ----- | ------------- | ---------------- | ----------------- | ---------------- |",
    ]
    for model_name, run_composite, avg_in, avg_out, p50 in rows:
        cells = [
            model_name,
            _format_cell(run_composite),
            _format_cell(avg_in),
            _format_cell(avg_out),
            _format_cell(p50),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render eval metrics summary as a Markdown results table.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to eval_metrics_summary.json (default: data/results/eval_metrics_summary.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write table to file (default: print to stdout)",
    )
    parser.add_argument(
        "--recommended",
        type=str,
        default=None,
        metavar="MODEL",
        help="Put this model first in the table (recommended model)",
    )
    args = parser.parse_args()

    summary_path = args.summary if args.summary is not None else _default_summary_path()
    if not summary_path.is_file():
        raise SystemExit(f"Summary file not found: {summary_path}")

    table = render_table(summary_path, recommended_model=args.recommended)
    if args.output is not None:
        args.output.write_text(table, encoding="utf-8")
        logger.info(f"Wrote table to {args.output}")
    else:
        logger.info(table)


if __name__ == "__main__":
    main()
