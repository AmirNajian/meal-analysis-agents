"""Eval runner: discover image–JSON pairs and run pipeline per sample."""

from meal_analysis.schemas import EvalSample
from evals.runner import discover_pairs

__all__ = [
    "EvalSample",
    "discover_pairs",
]
