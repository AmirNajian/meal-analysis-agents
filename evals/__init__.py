"""Eval runner: discover image–JSON pairs and run pipeline per sample."""

from meal_analysis.schemas import EvalSample
from evals.metrics import compute_metrics, p50_latency_ms, run_level_composite, score_guardrails, score_meal, score_safety
from evals.runner import (
    compute_metrics_from_file,
    discover_pairs,
    load_ground_truth,
    load_results,
    run_all,
    run_one,
    write_results,
)

__all__ = [
    "EvalSample",
    "compute_metrics",
    "compute_metrics_from_file",
    "discover_pairs",
    "load_ground_truth",
    "load_results",
    "p50_latency_ms",
    "run_all",
    "run_level_composite",
    "run_one",
    "score_guardrails",
    "score_meal",
    "score_safety",
    "write_results",
]
