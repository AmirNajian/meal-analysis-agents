"""Unit tests for meal_analysis.agents.safety_checks."""

from __future__ import annotations

import json
from typing import Any

import pytest

from meal_analysis.client import ChatCompletionResult
from meal_analysis.schemas import SafetyChecks
from meal_analysis.agents import AgentParseError
from meal_analysis.agents.safety_checks import safety_checks


class _DummyClient:
    """Simple stand-in for OpenAIClient with a configurable response."""

    def __init__(self, result: ChatCompletionResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:  # type: ignore[override]
        self.calls.append(kwargs)
        return self._result


@pytest.mark.asyncio
async def test_safety_checks_happy_path(safety_checks_dict: dict[str, Any]) -> None:
    """Valid JSON payload is parsed into SafetyChecks."""
    result = ChatCompletionResult(
        content=json.dumps(safety_checks_dict),
        input_tokens=5,
        output_tokens=10,
    )
    client = _DummyClient(result)

    parsed, in_tok, out_tok = await safety_checks(
        text="Eat more fiber. Choose whole grains.",
        client=client,  # type: ignore[arg-type]
        model="gpt-4o",
    )

    assert isinstance(parsed, SafetyChecks)
    assert parsed.no_insuline_guidance is True
    assert parsed.no_carb_content is True
    assert parsed.no_emotional_or_judgmental_language is True
    assert parsed.no_risky_ingredient_substitutions is True
    assert parsed.no_treatment_recommendation is True
    assert parsed.no_medical_diagnosis is True
    assert in_tok == 5
    assert out_tok == 10

    assert client.calls
    call = client.calls[0]
    assert call["model"] == "gpt-4o"
    assert isinstance(call["messages"], list)
    assert call["response_format"] == {"type": "json_object"}
    # User message should contain the input text
    user_msg = next(m for m in call["messages"] if m["role"] == "user")
    assert "Eat more fiber" in user_msg["content"]


@pytest.mark.asyncio
async def test_safety_checks_invalid_json_raises_agent_parse_error() -> None:
    """Non-JSON content from the model should raise AgentParseError."""
    result = ChatCompletionResult(
        content="NOT JSON",
        input_tokens=1,
        output_tokens=1,
    )
    client = _DummyClient(result)

    with pytest.raises(AgentParseError) as exc_info:
        await safety_checks(
            text="Some guidance.",
            client=client,  # type: ignore[arg-type]
            model="gpt-4o",
        )

    err = exc_info.value
    assert err.agent_name == "safety_checks"
    assert "not valid json" in str(err).lower()


@pytest.mark.asyncio
async def test_safety_checks_schema_mismatch_raises_agent_parse_error(
    safety_checks_dict: dict[str, Any],
) -> None:
    """Missing required fields should cause schema validation to fail."""
    invalid_payload = {
        k: v for k, v in safety_checks_dict.items() if k != "no_medical_diagnosis"
    }
    result = ChatCompletionResult(
        content=json.dumps(invalid_payload),
        input_tokens=1,
        output_tokens=1,
    )
    client = _DummyClient(result)

    with pytest.raises(AgentParseError) as exc_info:
        await safety_checks(
            text="Some guidance.",
            client=client,  # type: ignore[arg-type]
            model="gpt-4o",
        )

    err = exc_info.value
    assert err.agent_name == "safety_checks"
    assert "schema" in str(err).lower()
