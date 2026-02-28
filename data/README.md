# Dataset: Meal images and ground-truth JSON

This folder contains the curated dataset used for building and evaluating the meal-analysis pipeline.

## Layout

| Path | Contents |
|------|----------|
| `images/` | Meal photos (JPEG). One file per sample. |
| `json-files/` | Ground-truth JSON per image. One file per sample. |

## Pairing convention

Image and JSON are paired by **basename**: for each `images/<id>.jpeg` there is a corresponding `json-files/<id>.json`. The JSON `fileName` field matches the image filename (e.g. `upload_1749840444495_ce8fbfbc-988e-4ee6-9e52-d7ebdf4a58bc.jpeg`). Use the stem (filename without extension) to match an image to its ground truth.

There are **72 image–JSON pairs** in total.

## Ground-truth JSON structure

Each JSON file has:

- **`title`** – Short meal title.
- **`fileName`** – Image filename (for reference).
- **`guardrailCheck`** – Input guardrail labels (all booleans):
  - `is_food`, `no_pii`, `no_humans`, `no_captcha`
- **`safetyChecks`** – Output safety labels (all booleans):
  - `no_insuline_guidance`, `no_carb_content`, `no_emotional_or_judgmental_language`, `no_risky_ingredient_substitutions`, `no_treatment_recommendation`, `no_medical_diagnosis`
- **`mealAnalysis`** – Full meal inference:
  - `is_food` (bool), `recommendation` (`"green"` \| `"yellow"` \| `"orange"` \| `"red"`), `guidance_message`, `meal_title`, `meal_description` (strings)
  - `macros`: `calories`, `carbohydrates`, `fats`, `proteins` (numbers)
  - `ingredients`: list of `{ "name": string, "impact": "green" | "yellow" | "orange" | "red" }`

Evals compare model outputs to these fields (exact match for guardrails/safety; assignment-defined metrics for meal analysis).
