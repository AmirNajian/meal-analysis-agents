"""Meal inference agent: vision → structured MealAnalysis JSON.

This agent wraps a vision-capable OpenAI model and returns a structured
`MealAnalysis` object using JSON mode for reliable parsing.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from meal_analysis.client import OpenAIClient, image_bytes_to_data_url
from meal_analysis.schemas import MealAnalysis
from meal_analysis.agents.guardrail_check import AgentParseError


_SYSTEM_PROMPT = """You are a meal analysis assistant for people managing diabetes.

Given a single meal image, infer the following fields and return ONLY a single JSON
object matching this exact schema:

{
  "is_food": true | false,
  "recommendation": "green" | "yellow" | "orange" | "red",
  "guidance_message": string,
  "meal_title": string,
  "meal_description": string,
  "macros": {
    "calories": number,
    "carbohydrates": number,
    "fats": number,
    "proteins": number
  },
  "ingredients": [
    { "name": string, "impact": "green" | "yellow" | "orange" | "red" },
    ...
  ]
}

Guidelines:
- recommendation:
  - \"green\": generally low glycemic impact and high in fiber or protein.
  - \"yellow\": mixed impact; acceptable in moderation or with portion control.
  - \"orange\": higher glycemic impact or less ideal composition; caution advised.
  - \"red\": likely to cause large blood sugar spikes; generally not recommended.
- guidance_message, meal_title, and meal_description must be neutral in tone.
  Do not include emotional or judgmental language.
- Do NOT provide medical advice, diagnosis, or specific treatment recommendations.
- Macros are approximate per-serving estimates.

Return only the JSON object, with no prose before or after it.
"""


async def meal_analysis(
    *,
    image_bytes: bytes,
    client: OpenAIClient,
    model: str,
) -> MealAnalysis:
    """Run meal analysis on an image and return structured inference."""
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
                        "Analyze this meal image and return a JSON object strictly "
                        "following the specified schema."
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
            agent_name="meal_analysis",
            message="Response content was not valid JSON",
            raw_content=result.content,
        ) from exc

    try:
        return MealAnalysis.model_validate(payload)
    except ValidationError as exc:
        raise AgentParseError(
            agent_name="meal_analysis",
            message="Response JSON did not match MealAnalysis schema",
            raw_content=result.content,
        ) from exc


__all__ = [
    "meal_analysis",
]

