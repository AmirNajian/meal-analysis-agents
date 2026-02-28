"""Unit tests for meal_analysis.agents.meal_analysis."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from meal_analysis.client import ChatCompletionResult
from meal_analysis.schemas import MealAnalysis
from meal_analysis.agents import AgentParseError
from meal_analysis.agents.meal_analysis import meal_analysis


class _DummyClient:
    """Simple stand-in for OpenAIClient with a configurable response."""

    def __init__(self, result: ChatCompletionResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:  # type: ignore[override]
        self.calls.append(kwargs)
        return self._result


@pytest.mark.asyncio
async def test_meal_analysis_happy_path(meal_analysis_dict: dict[str, Any]) -> None:
    """Valid JSON payload is parsed into MealAnalysis."""
    result = ChatCompletionResult(
        content=json.dumps(meal_analysis_dict),
        input_tokens=10,
        output_tokens=15,
    )
    client = _DummyClient(result)

    out = await meal_analysis(
        image_bytes=b"fake-image",
        client=client,  # type: ignore[arg-type]
        model="gpt-4o",
    )

    assert isinstance(out, MealAnalysis)
    assert out.is_food is True
    assert out.recommendation in {"green", "yellow", "orange", "red"}
    assert out.meal_title == meal_analysis_dict["meal_title"]
    assert out.macros.calories == meal_analysis_dict["macros"]["calories"]
    assert len(out.ingredients) == len(meal_analysis_dict["ingredients"])

    # Ensure messages and response_format were passed to chat_completion
    assert client.calls, "chat_completion should have been called exactly once"
    call = client.calls[0]
    assert call["model"] == "gpt-4o"
    assert isinstance(call["messages"], list)
    assert call["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_meal_analysis_invalid_json_raises_agent_parse_error() -> None:
    """Non-JSON content from the model should raise AgentParseError."""
    result = ChatCompletionResult(
        content="NOT JSON",
        input_tokens=1,
        output_tokens=1,
    )
    client = _DummyClient(result)

    with pytest.raises(AgentParseError) as exc_info:
        await meal_analysis(
            image_bytes=b"fake",
            client=client,  # type: ignore[arg-type]
            model="gpt-4o",
        )

    err = exc_info.value
    assert err.agent_name == "meal_analysis"
    assert "not valid json" in str(err).lower()


@pytest.mark.asyncio
async def test_meal_analysis_schema_mismatch_raises_agent_parse_error(
    meal_analysis_dict: dict[str, Any],
) -> None:
    """Missing required fields should cause schema validation to fail."""
    # Remove a required field (e.g. macros) to force ValidationError in MealAnalysis
    invalid_payload = {k: v for k, v in meal_analysis_dict.items() if k != "macros"}
    result = ChatCompletionResult(
        content=json.dumps(invalid_payload),
        input_tokens=1,
        output_tokens=1,
    )
    client = _DummyClient(result)

    with pytest.raises(AgentParseError) as exc_info:
        await meal_analysis(
            image_bytes=b"fake",
            client=client,  # type: ignore[arg-type]
            model="gpt-4o",
        )

    err = exc_info.value
    assert err.agent_name == "meal_analysis"
    assert "schema" in str(err).lower()

