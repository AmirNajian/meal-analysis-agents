"""Pipeline: run guardrail → meal analysis → safety checks with short-circuit on failure."""

from __future__ import annotations

from meal_analysis.agents import guardrail_check, meal_analysis, safety_checks
from meal_analysis.client import OpenAIClient
from meal_analysis.schemas import AnalysisResponse, GuardrailCheck, MealAnalysis, SafetyChecks


class PipelineResult:
    """Result of a successful pipeline run: response plus aggregated token usage."""

    __slots__ = ("response", "input_tokens", "output_tokens")

    def __init__(
        self,
        response: AnalysisResponse,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.response = response
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


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
) -> PipelineResult:
    """Run guardrail → meal analysis → safety checks; short-circuit on failure.

    Returns
    -------
    PipelineResult
        response: combined AnalysisResponse; input_tokens/output_tokens aggregated
        from all three agent calls.

    Raises
    ------
    GuardrailRejection
        If any input guardrail check is False.
    SafetyRejection
        If any output safety check is False.
    """
    guardrail, g_in, g_out = await guardrail_check(
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

    meal, m_in, m_out = await meal_analysis(
        image_bytes=image_bytes,
        client=client,
        model=model,
    )

    text = f"{meal.guidance_message}\nTitle: {meal.meal_title}\nDescription: {meal.meal_description}"
    safety, s_in, s_out = await safety_checks(
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

    return PipelineResult(
        response=AnalysisResponse(
            guardrailCheck=guardrail,
            mealAnalysis=meal,
            safetyChecks=safety,
        ),
        input_tokens=g_in + m_in + s_in,
        output_tokens=g_out + m_out + s_out,
    )


__all__ = [
    "GuardrailRejection",
    "PipelineResult",
    "SafetyRejection",
    "run_analysis_pipeline",
]
