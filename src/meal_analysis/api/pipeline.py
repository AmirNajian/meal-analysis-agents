"""Pipeline: run guardrail → meal analysis → safety checks with short-circuit on failure."""

from __future__ import annotations

from meal_analysis.agents import guardrail_check, meal_analysis, safety_checks
from meal_analysis.client import OpenAIClient
from meal_analysis.schemas import AnalysisResponse, GuardrailCheck, MealAnalysis, SafetyChecks


class GuardrailRejection(Exception):
    """Raised when input guardrails fail (not food, PII, humans, or captcha)."""

    def __init__(self, message: str, guardrail: GuardrailCheck) -> None:
        super().__init__(message)
        self.guardrail = guardrail


class SafetyRejection(Exception):
    """Raised when output safety checks fail."""

    def __init__(self, message: str, safety: SafetyChecks) -> None:
        super().__init__(message)
        self.safety = safety


async def run_analysis_pipeline(
    *,
    image_bytes: bytes,
    client: OpenAIClient,
    model: str,
) -> AnalysisResponse:
    """Run guardrail → meal analysis → safety checks; short-circuit on failure.

    Raises
    ------
    GuardrailRejection
        If any input guardrail check is False.
    SafetyRejection
        If any output safety check is False.
    """
    guardrail = await guardrail_check(
        image_bytes=image_bytes,
        client=client,
        model=model,
    )
    if not guardrail.is_food:
        raise GuardrailRejection("Input guardrails failed: not food", guardrail)
    if not guardrail.no_pii:
        raise GuardrailRejection("Input guardrails failed: PII detected", guardrail)
    if not guardrail.no_humans:
        raise GuardrailRejection("Input guardrails failed: humans detected", guardrail)
    if not guardrail.no_captcha:
        raise GuardrailRejection("Input guardrails failed: captcha detected", guardrail)

    meal = await meal_analysis(
        image_bytes=image_bytes,
        client=client,
        model=model,
    )

    text = f"{meal.guidance_message}\nTitle: {meal.meal_title}\nDescription: {meal.meal_description}"
    safety = await safety_checks(
        text=text,
        client=client,
        model=model,
    )

    if not all([
        safety.no_insuline_guidance,
        safety.no_carb_content,
        safety.no_emotional_or_judgmental_language,
        safety.no_risky_ingredient_substitutions,
        safety.no_treatment_recommendation,
        safety.no_medical_diagnosis,
    ]):
        raise SafetyRejection("Output safety check failed", safety)

    return AnalysisResponse(
        guardrailCheck=guardrail,
        mealAnalysis=meal,
        safetyChecks=safety,
    )


__all__ = [
    "GuardrailRejection",
    "SafetyRejection",
    "run_analysis_pipeline",
]
