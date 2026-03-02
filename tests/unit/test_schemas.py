"""Unit tests for meal_analysis.schemas."""

import pytest
from pydantic import ValidationError

from meal_analysis.schemas import (
    AnalysisResponse,
    EvalSample,
    EvalSampleResult,
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


# ---- EvalSample ----
def test_eval_sample_sample_id(tmp_path) -> None:
    """EvalSample.sample_id is the image path stem."""
    img = tmp_path / "meal_abc.jpeg"
    js = tmp_path / "meal_abc.json"
    img.touch()
    js.touch()
    s = EvalSample(image_path=img, json_path=js)
    assert s.sample_id == "meal_abc"


# ---- EvalSampleResult ----
def test_eval_sample_result_parses_minimal() -> None:
    """EvalSampleResult accepts required fields and defaults for optional."""
    r = EvalSampleResult(sample_id="x", latency_ms=50.0, success=True)
    assert r.sample_id == "x"
    assert r.latency_ms == 50.0
    assert r.success is True
    assert r.response is None
    assert r.error_class is None
    assert r.input_tokens is None
    assert r.output_tokens is None
    assert r.guardrail_latency_ms is None
    assert r.meal_latency_ms is None
    assert r.safety_latency_ms is None


def test_eval_sample_result_rejects_negative_latency() -> None:
    with pytest.raises(ValidationError):
        EvalSampleResult(sample_id="x", latency_ms=-1.0, success=False)


def test_eval_sample_result_accepts_optional_tokens(analysis_response_dict: dict) -> None:
    """EvalSampleResult accepts input_tokens, output_tokens, per-agent latencies, and response."""
    resp = AnalysisResponse.model_validate(analysis_response_dict)
    r = EvalSampleResult(
        sample_id="y",
        latency_ms=100.0,
        success=True,
        response=resp,
        input_tokens=10,
        output_tokens=20,
        guardrail_latency_ms=50.0,
        meal_latency_ms=200.0,
        safety_latency_ms=80.0,
    )
    assert r.input_tokens == 10
    assert r.output_tokens == 20
    assert r.guardrail_latency_ms == 50.0
    assert r.meal_latency_ms == 200.0
    assert r.safety_latency_ms == 80.0
    assert r.response is not None
    assert r.response.mealAnalysis.meal_title == "Salad"


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
