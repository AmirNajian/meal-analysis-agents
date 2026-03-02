"""Unit tests for evals.metrics."""

from evals.metrics import (
    compute_metrics,
    p50_latency_ms,
    run_level_composite,
    score_guardrails,
    score_meal,
    score_safety,
)
from meal_analysis.schemas import (
    AnalysisResponse,
    EvalSampleResult,
    GroundTruthRecord,
    GuardrailCheck,
    Macros,
    MealAnalysis,
    SafetyChecks,
)
from meal_analysis.schemas import Ingredient


def _make_gt(
    guardrail: GuardrailCheck,
    safety: SafetyChecks,
    meal: MealAnalysis,
    sample_id: str = "s1",
) -> GroundTruthRecord:
    return GroundTruthRecord(
        title="Test",
        fileName=f"{sample_id}.jpeg",
        guardrailCheck=guardrail,
        safetyChecks=safety,
        mealAnalysis=meal,
    )


def _make_result(
    sample_id: str,
    response: AnalysisResponse | None,
    latency_ms: float = 50.0,
) -> EvalSampleResult:
    return EvalSampleResult(
        sample_id=sample_id,
        latency_ms=latency_ms,
        success=response is not None,
        response=response,
    )


def test_score_guardrails_perfect_match() -> None:
    """When all guardrail booleans match, score is 100."""
    g = GuardrailCheck(is_food=True, no_pii=True, no_humans=True, no_captcha=True)
    s = SafetyChecks(
        no_insuline_guidance=True,
        no_carb_content=True,
        no_emotional_or_judgmental_language=True,
        no_risky_ingredient_substitutions=True,
        no_treatment_recommendation=True,
        no_medical_diagnosis=True,
    )
    meal = MealAnalysis(
        is_food=True,
        recommendation="green",
        guidance_message="x",
        meal_title="x",
        meal_description="x",
        macros=Macros(calories=100, carbohydrates=10, fats=5, proteins=3),
        ingredients=[Ingredient(name="Lettuce", impact="green")],
    )
    gt = _make_gt(g, s, meal)
    resp = AnalysisResponse(guardrailCheck=g, mealAnalysis=meal, safetyChecks=s)
    results = [_make_result("s1", resp)]
    gt_by_id = {"s1": gt}
    assert score_guardrails(results, gt_by_id) == 100.0


def test_score_guardrails_mismatch() -> None:
    """When guardrail differs, score is 0 for that sample."""
    g_gt = GuardrailCheck(is_food=True, no_pii=True, no_humans=True, no_captcha=True)
    g_pred = GuardrailCheck(is_food=False, no_pii=True, no_humans=True, no_captcha=True)
    s = SafetyChecks(
        no_insuline_guidance=True,
        no_carb_content=True,
        no_emotional_or_judgmental_language=True,
        no_risky_ingredient_substitutions=True,
        no_treatment_recommendation=True,
        no_medical_diagnosis=True,
    )
    meal = MealAnalysis(
        is_food=True,
        recommendation="green",
        guidance_message="x",
        meal_title="x",
        meal_description="x",
        macros=Macros(calories=100, carbohydrates=10, fats=5, proteins=3),
        ingredients=[],
    )
    gt = _make_gt(g_gt, s, meal)
    resp = AnalysisResponse(guardrailCheck=g_pred, mealAnalysis=meal, safetyChecks=s)
    results = [_make_result("s1", resp)]
    gt_by_id = {"s1": gt}
    assert score_guardrails(results, gt_by_id) == 0.0


def test_score_safety_perfect_match() -> None:
    """When all safety booleans match, score is 100."""
    g = GuardrailCheck(is_food=True, no_pii=True, no_humans=True, no_captcha=True)
    s = SafetyChecks(
        no_insuline_guidance=True,
        no_carb_content=True,
        no_emotional_or_judgmental_language=True,
        no_risky_ingredient_substitutions=True,
        no_treatment_recommendation=True,
        no_medical_diagnosis=True,
    )
    meal = MealAnalysis(
        is_food=True,
        recommendation="green",
        guidance_message="x",
        meal_title="x",
        meal_description="x",
        macros=Macros(calories=100, carbohydrates=10, fats=5, proteins=3),
        ingredients=[],
    )
    gt = _make_gt(g, s, meal)
    resp = AnalysisResponse(guardrailCheck=g, mealAnalysis=meal, safetyChecks=s)
    results = [_make_result("s1", resp)]
    gt_by_id = {"s1": gt}
    assert score_safety(results, gt_by_id) == 100.0


def test_run_level_composite_formula() -> None:
    """Run composite = 0.2*g + 0.5*m + 0.3*s."""
    assert run_level_composite(100.0, 100.0, 100.0) == 100.0
    assert run_level_composite(0.0, 0.0, 0.0) == 0.0
    # 0.2*100 + 0.5*0 + 0.3*0 = 20
    assert run_level_composite(100.0, 0.0, 0.0) == 20.0


def test_p50_latency_ms() -> None:
    """P50 is median of latency_ms."""
    results = [
        EvalSampleResult(sample_id="a", latency_ms=10.0, success=False),
        EvalSampleResult(sample_id="b", latency_ms=20.0, success=False),
        EvalSampleResult(sample_id="c", latency_ms=30.0, success=False),
    ]
    assert p50_latency_ms(results) == 20.0
    assert p50_latency_ms([]) == 0.0


def test_compute_metrics_returns_all_keys() -> None:
    """compute_metrics returns dict with all metric keys including avg_input_tokens, avg_output_tokens."""
    g = GuardrailCheck(is_food=True, no_pii=True, no_humans=True, no_captcha=True)
    s = SafetyChecks(
        no_insuline_guidance=True,
        no_carb_content=True,
        no_emotional_or_judgmental_language=True,
        no_risky_ingredient_substitutions=True,
        no_treatment_recommendation=True,
        no_medical_diagnosis=True,
    )
    meal = MealAnalysis(
        is_food=True,
        recommendation="green",
        guidance_message="x",
        meal_title="x",
        meal_description="x",
        macros=Macros(calories=100, carbohydrates=10, fats=5, proteins=3),
        ingredients=[Ingredient(name="Lettuce", impact="green")],
    )
    gt = _make_gt(g, s, meal)
    resp = AnalysisResponse(guardrailCheck=g, mealAnalysis=meal, safetyChecks=s)
    results = [_make_result("s1", resp, latency_ms=75.0)]
    gt_by_id = {"s1": gt}
    m = compute_metrics(results, gt_by_id)
    assert set(m.keys()) == {
        "guardrails_pct", "safety_pct", "meal_pct", "run_composite", "p50_latency_ms",
        "avg_input_tokens", "avg_output_tokens",
    }
    assert m["guardrails_pct"] == 100.0
    assert m["safety_pct"] == 100.0
    assert m["run_composite"] == 100.0
    assert m["p50_latency_ms"] == 75.0
    assert m["avg_input_tokens"] is None
    assert m["avg_output_tokens"] is None


def test_compute_metrics_avg_tokens_when_present() -> None:
    """When results have input_tokens/output_tokens, compute_metrics returns their averages."""
    g = GuardrailCheck(is_food=True, no_pii=True, no_humans=True, no_captcha=True)
    s = SafetyChecks(
        no_insuline_guidance=True,
        no_carb_content=True,
        no_emotional_or_judgmental_language=True,
        no_risky_ingredient_substitutions=True,
        no_treatment_recommendation=True,
        no_medical_diagnosis=True,
    )
    meal = MealAnalysis(
        is_food=True,
        recommendation="green",
        guidance_message="x",
        meal_title="x",
        meal_description="x",
        macros=Macros(calories=100, carbohydrates=10, fats=5, proteins=3),
        ingredients=[Ingredient(name="Lettuce", impact="green")],
    )
    gt = _make_gt(g, s, meal)
    resp = AnalysisResponse(guardrailCheck=g, mealAnalysis=meal, safetyChecks=s)
    results = [
        EvalSampleResult(sample_id="s1", latency_ms=10.0, success=True, response=resp, input_tokens=100, output_tokens=50),
        EvalSampleResult(sample_id="s2", latency_ms=20.0, success=True, response=resp, input_tokens=200, output_tokens=100),
    ]
    gt_by_id = {"s1": gt, "s2": gt}
    m = compute_metrics(results, gt_by_id)
    assert m["avg_input_tokens"] == 150.0
    assert m["avg_output_tokens"] == 75.0


def test_meal_recommendation_orange_maps_to_red() -> None:
    """3-class: orange and red both count as same for recommendation match."""
    g = GuardrailCheck(is_food=True, no_pii=True, no_humans=True, no_captcha=True)
    s = SafetyChecks(
        no_insuline_guidance=True,
        no_carb_content=True,
        no_emotional_or_judgmental_language=True,
        no_risky_ingredient_substitutions=True,
        no_treatment_recommendation=True,
        no_medical_diagnosis=True,
    )
    meal_gt = MealAnalysis(
        is_food=True,
        recommendation="red",
        guidance_message="x",
        meal_title="x",
        meal_description="x",
        macros=Macros(calories=100, carbohydrates=10, fats=5, proteins=3),
        ingredients=[Ingredient(name="Lettuce", impact="green")],
    )
    meal_pred = MealAnalysis(
        is_food=True,
        recommendation="orange",  # 3-class: orange -> red, so matches gt "red"
        guidance_message="x",
        meal_title="x",
        meal_description="x",
        macros=Macros(calories=100, carbohydrates=10, fats=5, proteins=3),
        ingredients=[Ingredient(name="Lettuce", impact="green")],
    )
    gt = _make_gt(g, s, meal_gt)
    resp = AnalysisResponse(guardrailCheck=g, mealAnalysis=meal_pred, safetyChecks=s)
    results = [_make_result("s1", resp)]
    gt_by_id = {"s1": gt}
    # Meal score should be 100 for recommendation (orange vs red match in 3-class)
    m = compute_metrics(results, gt_by_id)
    assert m["meal_pct"] == 100.0
