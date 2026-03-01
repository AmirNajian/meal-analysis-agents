"""Collect metrics from per-model eval result JSONs into a single summary file.

Usage:
  uv run python -m evals.collect_metrics --output-prefix PREFIX [--results-dir DIR] --data-dir D --metrics-out OUT.json MODEL [MODEL ...]

Result files are looked up as <results_dir>/PREFIX_<slug>.json (default results-dir: current directory).
Uses the same slug logic as evals.runner for filenames.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from evals.runner import compute_metrics_from_file


def _sanitize_model_for_path(model: str) -> str:
    """Match runner's slug: filename-safe model id."""
    return model.replace("/", "-").replace(" ", "_")


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate eval metrics from per-model result JSONs into one file.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        metavar="PREFIX",
        help="Stem of result files (e.g. eval_results → eval_results_gpt-4o.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data directory (default: project root / data)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Override images directory (default: data_dir / images)",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Override json-files directory (default: data_dir / json-files)",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("eval_metrics_summary.json"),
        metavar="FILE",
        help="Output path for aggregated metrics JSON (default: eval_metrics_summary.json)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("."),
        metavar="DIR",
        help="Directory containing PREFIX_<model_slug>.json files (default: current directory)",
    )
    parser.add_argument(
        "models",
        nargs="+",
        type=str,
        metavar="MODEL",
        help="Model IDs to collect (same order/names as used with evals.runner --models)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else _default_data_dir()
    summary: dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_prefix": args.output_prefix,
        "data_dir": str(data_dir),
        "models": {},
    }

    results_dir = args.results_dir.resolve()
    for model in args.models:
        slug = _sanitize_model_for_path(model)
        result_path = results_dir / f"{args.output_prefix}_{slug}.json"
        if not result_path.is_file():
            print(f"Skip {model}: {result_path} not found")
            continue
        metrics = compute_metrics_from_file(
            result_path,
            data_dir=data_dir,
            images_dir=args.images_dir,
            json_dir=args.json_dir,
        )
        summary["models"][model] = metrics
        print(f"Collected {model}: run_composite={metrics['run_composite']}, p50_latency_ms={metrics['p50_latency_ms']}")

    args.metrics_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {len(summary['models'])} model(s) to {args.metrics_out}")


if __name__ == "__main__":
    main()
