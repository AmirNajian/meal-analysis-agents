"""Input guardrails agent: checks basic safety and suitability of an image.

This agent wraps a vision-capable OpenAI model and returns a structured
`GuardrailCheck` object using JSON mode for reliable parsing.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from meal_analysis.client import OpenAIClient, image_bytes_to_data_url
from meal_analysis.schemas import GuardrailCheck


class AgentParseError(RuntimeError):
    """Raised when an agent returns JSON that cannot be parsed or validated."""

    def __init__(self, agent_name: str, message: str, raw_content: str) -> None:
        super().__init__(f"{agent_name} parse error: {message}")
        self.agent_name = agent_name
        self.raw_content = raw_content


_SYSTEM_PROMPT = """You are an input guardrails classifier for a health-tech app.

Given a single meal-related image, you must answer four questions:
- is_food: whether the content of the image is food (true/false)
- no_pii: whether the image does NOT contain personally identifiable information
- no_humans: whether the image does NOT contain humans or visible body parts
- no_captcha: whether the image does NOT contain CAPTCHAs or similar verification prompts

Return ONLY a single JSON object with the exact keys:
  {
    "is_food": true | false,
    "no_pii": true | false,
    "no_humans": true | false,
    "no_captcha": true | false
  }

Do not include any explanations or extra fields.
"""


async def guardrail_check(
    *,
    image_bytes: bytes,
    client: OpenAIClient,
    model: str,
) -> tuple[GuardrailCheck, int, int]:
    """Run input guardrails on an image and return structured booleans plus token usage.

    Returns
    -------
    tuple of (GuardrailCheck, input_tokens, output_tokens)
    """
    data_url = image_bytes_to_data_url(image_bytes, media_type="image/jpeg")

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": _SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this image and answer the four boolean fields as "
                        "described in the instructions."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        },
    ]

    result = await client.chat_completion(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise AgentParseError(
            agent_name="guardrail_check",
            message="Response content was not valid JSON",
            raw_content=result.content,
        ) from exc

    try:
        parsed = GuardrailCheck.model_validate(payload)
        return (parsed, result.input_tokens, result.output_tokens)
    except ValidationError as exc:
        raise AgentParseError(
            agent_name="guardrail_check",
            message="Response JSON did not match GuardrailCheck schema",
            raw_content=result.content,
        ) from exc


__all__ = [
    "AgentParseError",
    "guardrail_check",
]

