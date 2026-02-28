"""Pytest fixtures for tests/unit (schema and other unit tests)."""

from __future__ import annotations

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
