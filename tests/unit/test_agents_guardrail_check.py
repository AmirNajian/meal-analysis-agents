"""Unit tests for meal_analysis.agents.guardrail_check."""

from __future__ import annotations

import json
from typing import Any

import pytest

from meal_analysis.client import ChatCompletionResult
from meal_analysis.schemas import GuardrailCheck
from meal_analysis.agents.guardrail_check import AgentParseError, guardrail_check


class _DummyClient:
    """Simple stand-in for OpenAIClient with a configurable response."""

    def __init__(self, result: ChatCompletionResult) -> None:
        self._result = result
        self.calls: list[dict[str, Any]] = []

    async def chat_completion(self, **kwargs: Any) -> ChatCompletionResult:  # type: ignore[override]
        self.calls.append(kwargs)
        return self._result


@pytest.mark.asyncio
async def test_guardrail_check_happy_path(guardrail_check_dict: dict[str, Any]) -> None:
    """Valid JSON payload is parsed into GuardrailCheck."""
    result = ChatCompletionResult(
        content=json.dumps(guardrail_check_dict),
        input_tokens=5,
        output_tokens=3,
    )
    client = _DummyClient(result)

    parsed, in_tok, out_tok = await guardrail_check(
        image_bytes=b"fake-bytes",
        client=client,  # type: ignore[arg-type]
        model="gpt-4o",
    )

    assert isinstance(parsed, GuardrailCheck)
    assert parsed.is_food is True
    assert parsed.no_pii is True
    assert parsed.no_humans is True
    assert parsed.no_captcha is True
    assert in_tok == 5
    assert out_tok == 3

    # Ensure messages and response_format were passed to chat_completion
    assert client.calls, "chat_completion should have been called exactly once"
    call = client.calls[0]
    assert call["model"] == "gpt-4o"
    assert isinstance(call["messages"], list)
    assert call["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_guardrail_check_invalid_json_raises_agent_parse_error() -> None:
    """Non-JSON content from the model should raise AgentParseError."""
    result = ChatCompletionResult(
        content="NOT JSON",
        input_tokens=1,
        output_tokens=1,
    )
    client = _DummyClient(result)

    with pytest.raises(AgentParseError) as exc_info:
        await guardrail_check(
            image_bytes=b"fake",
            client=client,  # type: ignore[arg-type]
            model="gpt-4o",
        )

    err = exc_info.value
    assert err.agent_name == "guardrail_check"
    # Error message should indicate invalid JSON; exact casing is not important.
    assert "not valid json" in str(err).lower()


@pytest.mark.asyncio
async def test_guardrail_check_schema_mismatch_raises_agent_parse_error(
    guardrail_check_dict: dict[str, Any],
) -> None:
    """Missing required fields should cause schema validation to fail."""
    # Remove a required field to force ValidationError in GuardrailCheck
    invalid_payload = {k: v for k, v in guardrail_check_dict.items() if k != "no_captcha"}
    result = ChatCompletionResult(
        content=json.dumps(invalid_payload),
        input_tokens=1,
        output_tokens=1,
    )
    client = _DummyClient(result)

    with pytest.raises(AgentParseError) as exc_info:
        await guardrail_check(
            image_bytes=b"fake",
            client=client,  # type: ignore[arg-type]
            model="gpt-4o",
        )

    err = exc_info.value
    assert err.agent_name == "guardrail_check"
    assert "schema" in str(err).lower()

