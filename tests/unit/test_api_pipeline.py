"""Unit tests for meal_analysis.api.pipeline."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meal_analysis.api.pipeline import (
    GuardrailRejection,
    PipelineResult,
    SafetyRejection,
    run_analysis_pipeline,
)
from meal_analysis.schemas import AnalysisResponse


@pytest.fixture
def mock_client() -> MagicMock:
    """Dummy client for pipeline (agents are mocked, client not used for HTTP)."""
    return MagicMock()


@pytest.mark.asyncio
async def test_run_analysis_pipeline_success(
    mock_client: MagicMock,
    pipeline_image_bytes: bytes,
    guardrail_check_passed,
    meal_analysis_result,
    safety_checks_passed,
) -> None:
    """When all agents pass, pipeline returns PipelineResult with response and token counts."""
    with (
        patch("meal_analysis.api.pipeline.guardrail_check", new_callable=AsyncMock) as m_guard,
        patch("meal_analysis.api.pipeline.meal_analysis", new_callable=AsyncMock) as m_meal,
        patch("meal_analysis.api.pipeline.safety_checks", new_callable=AsyncMock) as m_safety,
    ):
        m_guard.return_value = guardrail_check_passed
        m_meal.return_value = meal_analysis_result
        m_safety.return_value = safety_checks_passed

        result = await run_analysis_pipeline(
            image_bytes=pipeline_image_bytes,
            client=mock_client,
            model="gpt-4o",
        )

    assert isinstance(result, PipelineResult)
    assert result.response.guardrailCheck is guardrail_check_passed
    assert result.response.mealAnalysis is meal_analysis_result
    assert result.response.safetyChecks is safety_checks_passed
    assert result.input_tokens == 0
    assert result.output_tokens == 0

    m_guard.assert_called_once()
    m_meal.assert_called_once()
    m_safety.assert_called_once()
    # safety_checks receives concatenated text from meal
    call_kw = m_safety.call_args.kwargs
    assert "Limit this meal." in call_kw["text"]
    assert "Plain Rolls" in call_kw["text"]


@pytest.mark.asyncio
async def test_run_analysis_pipeline_success_returns_pipeline_result_with_token_fields(
    mock_client: MagicMock,
    pipeline_image_bytes: bytes,
    guardrail_check_passed,
    meal_analysis_result,
    safety_checks_passed,
) -> None:
    """Pipeline success returns PipelineResult with response, input_tokens, and output_tokens (tokens 0 until agents expose usage)."""
    with (
        patch("meal_analysis.api.pipeline.guardrail_check", new_callable=AsyncMock) as m_guard,
        patch("meal_analysis.api.pipeline.meal_analysis", new_callable=AsyncMock) as m_meal,
        patch("meal_analysis.api.pipeline.safety_checks", new_callable=AsyncMock) as m_safety,
    ):
        m_guard.return_value = guardrail_check_passed
        m_meal.return_value = meal_analysis_result
        m_safety.return_value = safety_checks_passed

        result = await run_analysis_pipeline(
            image_bytes=pipeline_image_bytes,
            client=mock_client,
            model="gpt-4o",
        )

    assert isinstance(result, PipelineResult)
    assert hasattr(result, "response")
    assert hasattr(result, "input_tokens")
    assert hasattr(result, "output_tokens")
    assert isinstance(result.response, AnalysisResponse)
    assert isinstance(result.input_tokens, int) and result.input_tokens >= 0
    assert isinstance(result.output_tokens, int) and result.output_tokens >= 0


@pytest.mark.asyncio
async def test_run_analysis_pipeline_guardrail_not_food_raises(
    mock_client: MagicMock,
    pipeline_image_bytes: bytes,
    guardrail_check_not_food,
) -> None:
    """When guardrail reports not food, pipeline raises GuardrailRejection and does not call meal or safety."""
    with (
        patch("meal_analysis.api.pipeline.guardrail_check", new_callable=AsyncMock) as m_guard,
        patch("meal_analysis.api.pipeline.meal_analysis", new_callable=AsyncMock) as m_meal,
        patch("meal_analysis.api.pipeline.safety_checks", new_callable=AsyncMock) as m_safety,
    ):
        m_guard.return_value = guardrail_check_not_food

        with pytest.raises(GuardrailRejection) as exc_info:
            await run_analysis_pipeline(
                image_bytes=pipeline_image_bytes,
                client=mock_client,
                model="gpt-4o",
            )

    err = exc_info.value
    assert "not food" in err.args[0]
    assert err.guardrail is guardrail_check_not_food
    m_guard.assert_called_once()
    m_meal.assert_not_called()
    m_safety.assert_not_called()


@pytest.mark.asyncio
async def test_run_analysis_pipeline_safety_failure_raises(
    mock_client: MagicMock,
    pipeline_image_bytes: bytes,
    guardrail_check_passed,
    meal_analysis_result,
    safety_checks_failed,
) -> None:
    """When safety checks fail, pipeline raises SafetyRejection."""
    with (
        patch("meal_analysis.api.pipeline.guardrail_check", new_callable=AsyncMock) as m_guard,
        patch("meal_analysis.api.pipeline.meal_analysis", new_callable=AsyncMock) as m_meal,
        patch("meal_analysis.api.pipeline.safety_checks", new_callable=AsyncMock) as m_safety,
    ):
        m_guard.return_value = guardrail_check_passed
        m_meal.return_value = meal_analysis_result
        m_safety.return_value = safety_checks_failed

        with pytest.raises(SafetyRejection) as exc_info:
            await run_analysis_pipeline(
                image_bytes=pipeline_image_bytes,
                client=mock_client,
                model="gpt-4o",
            )

    err = exc_info.value
    assert "safety" in err.args[0].lower()
    assert err.safety is safety_checks_failed
    m_guard.assert_called_once()
    m_meal.assert_called_once()
    m_safety.assert_called_once()
