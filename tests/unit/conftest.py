"""Pytest fixtures for tests/unit (schema and other unit tests)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def guardrail_check_dict() -> dict[str, Any]:
    """Valid guardrailCheck payload."""
    return {"is_food": True, "no_pii": True, "no_humans": True, "no_captcha": True}


@pytest.fixture
def safety_checks_dict() -> dict[str, Any]:
    """Valid safetyChecks payload."""
    return {
        "no_insuline_guidance": True,
        "no_carb_content": True,
        "no_emotional_or_judgmental_language": True,
        "no_risky_ingredient_substitutions": True,
        "no_treatment_recommendation": True,
        "no_medical_diagnosis": True,
    }


@pytest.fixture
def macros_dict() -> dict[str, Any]:
    """Valid macros payload."""
    return {"calories": 270, "carbohydrates": 54, "fats": 3, "proteins": 7}


@pytest.fixture
def ingredient_dict() -> dict[str, Any]:
    """Valid single ingredient payload."""
    return {"name": "Lettuce", "impact": "green"}


@pytest.fixture
def meal_analysis_dict(
    macros_dict: dict[str, Any],
) -> dict[str, Any]:
    """Valid mealAnalysis payload (uses macros_dict)."""
    return {
        "is_food": True,
        "recommendation": "orange",
        "guidance_message": "Limit this meal.",
        "meal_title": "Plain Rolls",
        "meal_description": "White bread rolls.",
        "macros": macros_dict,
        "ingredients": [
            {"name": "Flour", "impact": "orange"},
            {"name": "Water", "impact": "green"},
        ],
    }


@pytest.fixture
def analysis_response_dict(
    guardrail_check_dict: dict[str, Any],
    safety_checks_dict: dict[str, Any],
) -> dict[str, Any]:
    """Valid AnalysisResponse payload (guardrail + meal + safety)."""
    return {
        "guardrailCheck": guardrail_check_dict,
        "mealAnalysis": {
            "is_food": True,
            "recommendation": "green",
            "guidance_message": "Good choice.",
            "meal_title": "Salad",
            "meal_description": "Green salad.",
            "macros": {"calories": 100, "carbohydrates": 10, "fats": 5, "proteins": 3},
            "ingredients": [{"name": "Lettuce", "impact": "green"}],
        },
        "safetyChecks": safety_checks_dict,
    }


@pytest.fixture
def ground_truth_record_dict(
    guardrail_check_dict: dict[str, Any],
    safety_checks_dict: dict[str, Any],
) -> dict[str, Any]:
    """Valid GroundTruthRecord payload (shape of data/json-files/*.json)."""
    return {
        "title": "Plain Bread Rolls",
        "fileName": "upload_1749840444495_ce8fbfbc-988e-4ee6-9e52-d7ebdf4a58bc.jpeg",
        "guardrailCheck": guardrail_check_dict,
        "safetyChecks": safety_checks_dict,
        "mealAnalysis": {
            "is_food": True,
            "recommendation": "orange",
            "guidance_message": "Limit this meal.",
            "meal_title": "Plain Bread Rolls",
            "meal_description": "Three soft white bread rolls.",
            "macros": {"calories": 270, "carbohydrates": 54, "fats": 3, "proteins": 7},
            "ingredients": [
                {"name": "Refined wheat flour", "impact": "orange"},
                {"name": "Water", "impact": "green"},
            ],
        },
    }


# ---- Client (OpenAI) fixtures ----

@pytest.fixture
def openai_chat_response(openai_chat_content: str) -> dict[str, Any]:
    """Minimal successful OpenAI chat completions response with usage."""
    return {
        "choices": [
            {"message": {"content": openai_chat_content, "role": "assistant"}, "index": 0}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


@pytest.fixture
def openai_chat_content() -> str:
    """Default content string for mocked OpenAI chat response."""
    return '{"is_food": true, "no_pii": true}'


@pytest.fixture
def openai_chat_response_no_usage(openai_chat_content: str) -> dict[str, Any]:
    """OpenAI response without usage (client should default tokens to 0)."""
    return {
        "choices": [
            {"message": {"content": openai_chat_content, "role": "assistant"}, "index": 0}
        ],
    }


@pytest.fixture
def openai_chat_response_empty_choices() -> dict[str, Any]:
    """OpenAI response with no choices (client should raise ValueError)."""
    return {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 0}}


# ---- Pipeline (api/pipeline) fixtures ----

@pytest.fixture
def pipeline_image_bytes() -> bytes:
    """Sample image bytes for pipeline tests (no real image needed)."""
    return b"\xff\xd8\xff fake jpeg bytes"


@pytest.fixture
def guardrail_check_passed(guardrail_check_dict: dict[str, Any]) -> "GuardrailCheck":
    """GuardrailCheck instance with all checks passed."""
    from meal_analysis.schemas import GuardrailCheck
    return GuardrailCheck.model_validate(guardrail_check_dict)


@pytest.fixture
def guardrail_check_not_food() -> "GuardrailCheck":
    """GuardrailCheck with is_food=False (triggers short-circuit)."""
    from meal_analysis.schemas import GuardrailCheck
    return GuardrailCheck(
        is_food=False,
        no_pii=True,
        no_humans=True,
        no_captcha=True,
    )


@pytest.fixture
def meal_analysis_result(meal_analysis_dict: dict[str, Any]) -> "MealAnalysis":
    """MealAnalysis instance for pipeline success path."""
    from meal_analysis.schemas import MealAnalysis
    return MealAnalysis.model_validate(meal_analysis_dict)


@pytest.fixture
def safety_checks_passed(safety_checks_dict: dict[str, Any]) -> "SafetyChecks":
    """SafetyChecks instance with all checks passed."""
    from meal_analysis.schemas import SafetyChecks
    return SafetyChecks.model_validate(safety_checks_dict)


@pytest.fixture
def safety_checks_failed() -> "SafetyChecks":
    """SafetyChecks with one check failed (triggers SafetyRejection)."""
    from meal_analysis.schemas import SafetyChecks
    return SafetyChecks(
        no_insuline_guidance=True,
        no_carb_content=True,
        no_emotional_or_judgmental_language=True,
        no_risky_ingredient_substitutions=True,
        no_treatment_recommendation=True,
        no_medical_diagnosis=False,
    )


# ---- Evals runner (discover_pairs) fixtures ----

# Minimal valid ground-truth JSON content for fixture files
_EVALS_GROUND_TRUTH_JSON = """{"title":"Fixture Meal","fileName":"meal_a.jpeg","guardrailCheck":{"is_food":true,"no_pii":true,"no_humans":true,"no_captcha":true},"safetyChecks":{"no_insuline_guidance":true,"no_carb_content":true,"no_emotional_or_judgmental_language":true,"no_risky_ingredient_substitutions":true,"no_treatment_recommendation":true,"no_medical_diagnosis":true},"mealAnalysis":{"is_food":true,"recommendation":"green","guidance_message":"OK","meal_title":"Fixture","meal_description":"A fixture meal.","macros":{"calories":100,"carbohydrates":10,"fats":5,"proteins":3},"ingredients":[{"name":"Lettuce","impact":"green"}]}}"""


@pytest.fixture
def evals_fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary directory with image and JSON pairs for discover_pairs tests.

    Layout:
      - images/meal_a.jpeg, meal_b.jpg, orphan.jpeg (no matching JSON)
      - images/meal_a.jpg (same stem as meal_a.jpeg; tests dedupe)
      - json-files/meal_a.json, meal_b.json

    Expect discover_pairs to return 2 pairs (meal_a, meal_b), sorted by sample_id.
    """
    root = tmp_path_factory.mktemp("evals_data")
    images_dir = root / "images"
    json_dir = root / "json-files"
    images_dir.mkdir()
    json_dir.mkdir()

    # Dummy image bytes (minimal valid JPEG-like prefix)
    image_bytes = b"\xff\xd8\xff fixture"

    (images_dir / "meal_a.jpeg").write_bytes(image_bytes)
    (images_dir / "meal_a.jpg").write_bytes(image_bytes)  # same stem, for dedupe
    (images_dir / "meal_b.jpg").write_bytes(image_bytes)
    (images_dir / "orphan.jpeg").write_bytes(image_bytes)  # no matching JSON

    (json_dir / "meal_a.json").write_text(_EVALS_GROUND_TRUTH_JSON, encoding="utf-8")
    (json_dir / "meal_b.json").write_text(_EVALS_GROUND_TRUTH_JSON.replace("meal_a", "meal_b"), encoding="utf-8")

    return root
