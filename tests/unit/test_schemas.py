"""Unit tests for meal_analysis.schemas."""

import pytest
from pydantic import ValidationError

from meal_analysis.schemas import (
    AnalysisResponse,
    GroundTruthRecord,
    GuardrailCheck,
    Ingredient,
    Macros,
    MealAnalysis,
    SafetyChecks,
)


# ---- GuardrailCheck ----
def test_guardrail_check_parses_valid(guardrail_check_dict: dict) -> None:
    r = GuardrailCheck.model_validate(guardrail_check_dict)
    assert r.is_food is True
    assert r.no_pii is True
    assert r.no_humans is True
    assert r.no_captcha is True


def test_guardrail_check_rejects_missing_field() -> None:
    with pytest.raises(ValidationError):
        GuardrailCheck.model_validate({"is_food": True, "no_pii": True, "no_humans": True})


def test_guardrail_check_rejects_wrong_type(guardrail_check_dict: dict) -> None:
    data = {**guardrail_check_dict, "is_food": []}
    with pytest.raises(ValidationError):
        GuardrailCheck.model_validate(data)


# ---- SafetyChecks ----
def test_safety_checks_parses_valid(safety_checks_dict: dict) -> None:
    s = SafetyChecks.model_validate(safety_checks_dict)
    assert s.no_insuline_guidance is True
    assert s.no_medical_diagnosis is True


def test_safety_checks_rejects_extra_ignored_or_strict(safety_checks_dict: dict) -> None:
    data = {**safety_checks_dict, "unknown_field": 1}
    s = SafetyChecks.model_validate(data)
    assert s.no_medical_diagnosis is True


# ---- Macros ----
def test_macros_parses_valid(macros_dict: dict) -> None:
    m = Macros.model_validate(macros_dict)
    assert m.calories == 270
    assert m.carbohydrates == 54
    assert m.fats == 3
    assert m.proteins == 7


def test_macros_coerces_int_to_float() -> None:
    data = {"calories": 100, "carbohydrates": 20, "fats": 5, "proteins": 4}
    m = Macros.model_validate(data)
    assert m.calories == 100.0
    assert m.carbohydrates == 20.0


# ---- Ingredient ----
def test_ingredient_parses_valid(ingredient_dict: dict) -> None:
    i = Ingredient.model_validate(ingredient_dict)
    assert i.name == "Lettuce"
    assert i.impact == "green"


@pytest.mark.parametrize("impact", ["green", "yellow", "orange", "red"])
def test_ingredient_accepts_all_recommendation_levels(impact: str) -> None:
    i = Ingredient.model_validate({"name": "x", "impact": impact})
    assert i.impact == impact


def test_ingredient_rejects_invalid_impact() -> None:
    with pytest.raises(ValidationError):
        Ingredient.model_validate({"name": "x", "impact": "purple"})


# ---- MealAnalysis ----
def test_meal_analysis_parses_valid(meal_analysis_dict: dict) -> None:
    m = MealAnalysis.model_validate(meal_analysis_dict)
    assert m.is_food is True
    assert m.recommendation == "orange"
    assert m.meal_title == "Plain Rolls"
    assert m.macros.calories == 270
    assert len(m.ingredients) == 2
    assert m.ingredients[0].name == "Flour" and m.ingredients[0].impact == "orange"


def test_meal_analysis_rejects_invalid_recommendation() -> None:
    data = {
        "is_food": True,
        "recommendation": "bad",
        "guidance_message": "x",
        "meal_title": "x",
        "meal_description": "x",
        "macros": {"calories": 0, "carbohydrates": 0, "fats": 0, "proteins": 0},
        "ingredients": [],
    }
    with pytest.raises(ValidationError):
        MealAnalysis.model_validate(data)


# ---- AnalysisResponse ----
def test_analysis_response_parses_valid(analysis_response_dict: dict) -> None:
    r = AnalysisResponse.model_validate(analysis_response_dict)
    assert r.guardrailCheck.is_food is True
    assert r.mealAnalysis.meal_title == "Salad"
    assert r.safetyChecks.no_medical_diagnosis is True


# ---- GroundTruthRecord ----
def test_ground_truth_record_parses_real_shape(ground_truth_record_dict: dict) -> None:
    """Parse a payload matching data/json-files/*.json."""
    r = GroundTruthRecord.model_validate(ground_truth_record_dict)
    assert r.title == "Plain Bread Rolls"
    assert "upload_1749840444495" in r.fileName
    assert r.guardrailCheck.no_captcha is True
    assert r.mealAnalysis.recommendation == "orange"
    assert len(r.mealAnalysis.ingredients) == 2


def test_models_are_frozen(guardrail_check_dict: dict) -> None:
    r = GuardrailCheck.model_validate(guardrail_check_dict)
    with pytest.raises((ValidationError, TypeError)):
        r.is_food = False  # type: ignore[misc]
