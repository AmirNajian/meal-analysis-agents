"""Eval runner: discover image–JSON pairs and run pipeline per sample."""

from meal_analysis.schemas import EvalSample
from evals.runner import discover_pairs, load_ground_truth, run_all, run_one, write_results

__all__ = [
    "EvalSample",
    "discover_pairs",
    "load_ground_truth",
    "run_all",
    "run_one",
    "write_results",
]
