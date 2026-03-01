"""Eval runner: discover image–JSON pairs and run pipeline per sample."""

from meal_analysis.schemas import EvalSample
from evals.metrics import compute_metrics, p50_latency_ms, run_level_composite, score_guardrails, score_meal, score_safety
from evals.runner import discover_pairs, load_ground_truth, run_all, run_one, write_results

__all__ = [
    "EvalSample",
    "compute_metrics",
    "discover_pairs",
    "load_ground_truth",
    "p50_latency_ms",
    "run_all",
    "run_level_composite",
    "run_one",
    "score_guardrails",
    "score_meal",
    "score_safety",
    "write_results",
]
