"""Pydantic schemas for guardrails, meal analysis, safety, and pipeline response."""

from typing import Literal

from pydantic import BaseModel, ConfigDict

# Shared literal for recommendation and ingredient impact (assignment: green | yellow | orange | red).
RecommendationLevel = Literal["green", "yellow", "orange", "red"]


class GuardrailCheck(BaseModel):
    """Input guardrails: image checks (is_food, no PII/humans/captcha)."""

    model_config = ConfigDict(frozen=True)

    is_food: bool
    no_pii: bool
    no_humans: bool
    no_captcha: bool


class SafetyChecks(BaseModel):
    """Output guardrails: safety checks on meal analysis text."""

    model_config = ConfigDict(frozen=True)

    no_insuline_guidance: bool
    no_carb_content: bool
    no_emotional_or_judgmental_language: bool
    no_risky_ingredient_substitutions: bool
    no_treatment_recommendation: bool
    no_medical_diagnosis: bool


class Macros(BaseModel):
    """Macro-nutrient estimates for a meal."""

    model_config = ConfigDict(frozen=True)

    calories: float
    carbohydrates: float
    fats: float
    proteins: float


class Ingredient(BaseModel):
    """Single ingredient with glycemic impact."""

    model_config = ConfigDict(frozen=True)

    name: str
    impact: RecommendationLevel


class MealAnalysis(BaseModel):
    """Full meal inference: recommendation, guidance, macros, ingredients."""

    is_food: bool
    recommendation: RecommendationLevel
    guidance_message: str
    meal_title: str
    meal_description: str
    macros: Macros
    ingredients: list[Ingredient]


class AnalysisResponse(BaseModel):
    """Successful pipeline response: all three agent outputs (API success payload)."""

    guardrailCheck: GuardrailCheck
    mealAnalysis: MealAnalysis
    safetyChecks: SafetyChecks


class GroundTruthRecord(BaseModel):
    """Full shape of ground-truth JSON files (data/json-files/*.json) for evals."""

    title: str
    fileName: str
    guardrailCheck: GuardrailCheck
    safetyChecks: SafetyChecks
    mealAnalysis: MealAnalysis


__all__ = [
    "AnalysisResponse",
    "GroundTruthRecord",
    "GuardrailCheck",
    "Ingredient",
    "Macros",
    "MealAnalysis",
    "RecommendationLevel",
    "SafetyChecks",
]
