"""Output guardrails agent: validates meal analysis text for safety.

This agent evaluates user-facing meal analysis text for prohibited content
(medical advice, emotional language, risky substitutions, etc.) and returns
a structured `SafetyChecks` object using JSON mode for reliable parsing.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from meal_analysis.client import OpenAIClient
from meal_analysis.schemas import SafetyChecks
from meal_analysis.agents.guardrail_check import AgentParseError


_SYSTEM_PROMPT = """You are an output safety classifier for a health-tech app.

You must evaluate meal guidance text and answer six yes/no questions. For each
question, answer true if the text does NOT contain the described content,
and false if it does:

- no_insuline_guidance: true if the text does NOT advise insulin changes or dosing.
- no_carb_content: true if the text does NOT specify precise carb counts or exchanges.
- no_emotional_or_judgmental_language: true if the text has no emotional or judgmental phrasing.
- no_risky_ingredient_substitutions: true if the text does NOT suggest risky swaps
  (e.g. insulin changes, medication swaps).
- no_treatment_recommendation: true if the text does NOT recommend treatments.
- no_medical_diagnosis: true if the text does NOT include medical diagnosis.

Return ONLY a single JSON object with these exact keys, all booleans:
  {
    "no_insuline_guidance": true | false,
    "no_carb_content": true | false,
    "no_emotional_or_judgmental_language": true | false,
    "no_risky_ingredient_substitutions": true | false,
    "no_treatment_recommendation": true | false,
    "no_medical_diagnosis": true | false
  }

Do not include any explanations or extra fields.
"""


async def safety_checks(
    *,
    text: str,
    client: OpenAIClient,
    model: str,
) -> SafetyChecks:
    """Run output safety checks on meal guidance text and return structured booleans.

    Parameters
    ----------
    text :
        Concatenated meal analysis text to evaluate (e.g. guidance_message +
        meal_title + meal_description).
    client :
        Shared OpenAIClient instance for making chat calls.
    model :
        OpenAI model name (e.g. gpt-4o or gpt-4o-mini).
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Evaluate the following meal guidance text and return a JSON object "
                "with the six boolean safety fields as described in the instructions.\n\n"
                f"---\n{text}\n---"
            ),
        },
    ]

    result = await client.chat_completion(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
    )

    try:
        payload = json.loads(result.content)
    except json.JSONDecodeError as exc:
        raise AgentParseError(
            agent_name="safety_checks",
            message="Response content was not valid JSON",
            raw_content=result.content,
        ) from exc

    try:
        return SafetyChecks.model_validate(payload)
    except ValidationError as exc:
        raise AgentParseError(
            agent_name="safety_checks",
            message="Response JSON did not match SafetyChecks schema",
            raw_content=result.content,
        ) from exc


__all__ = ["safety_checks"]
