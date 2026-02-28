"""Unit tests for POST /analyze endpoint (meal_analysis.api.main)."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from meal_analysis.api.main import app
from meal_analysis.api.pipeline import GuardrailRejection, SafetyRejection, run_analysis_pipeline
from meal_analysis.agents.guardrail_check import AgentParseError
from meal_analysis.schemas import AnalysisResponse

# So get_config() and OpenAIClient() succeed when TestClient runs the real lifespan
os.environ.setdefault("OPENAI_API_KEY", "test-key-for-unit-tests")


@pytest.fixture
def client() -> TestClient:
    """TestClient; uses real lifespan (env has OPENAI_API_KEY). Pipeline is mocked in tests."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_image_bytes() -> bytes:
    """Small image payload for successful requests."""
    return b"\xff\xd8\xff fake jpeg"


def test_analyze_success(
    client: TestClient,
    valid_image_bytes: bytes,
    guardrail_check_passed,
    meal_analysis_result,
    safety_checks_passed,
) -> None:
    """POST /analyze with valid file returns 200 and AnalysisResponse."""
    with patch("meal_analysis.api.main.run_analysis_pipeline", new_callable=AsyncMock) as m_run:
        m_run.return_value = AnalysisResponse(
            guardrailCheck=guardrail_check_passed,
            mealAnalysis=meal_analysis_result,
            safetyChecks=safety_checks_passed,
        )
        response = client.post(
            "/analyze",
            files={"file": ("meal.jpg", valid_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 200
    data = response.json()
    assert "guardrailCheck" in data
    assert "mealAnalysis" in data
    assert "safetyChecks" in data
    assert data["guardrailCheck"]["is_food"] is True
    m_run.assert_called_once()


def test_analyze_missing_filename_returns_422(client: TestClient) -> None:
    """POST /analyze with invalid file (e.g. empty filename) returns 422 (FastAPI validation)."""
    response = client.post(
        "/analyze",
        files={"file": ("", b"bytes", "image/jpeg")},
    )
    assert response.status_code == 422
    detail = response.json().get("detail", [])
    assert isinstance(detail, list) and len(detail) >= 1
    assert any("file" in str(e.get("loc", [])).lower() for e in detail if isinstance(e, dict))


def test_analyze_image_too_large_returns_400(client: TestClient) -> None:
    """POST /analyze with body larger than MAX_IMAGE_BYTES returns 400."""
    with patch("meal_analysis.api.main.run_analysis_pipeline", new_callable=AsyncMock):
        # MAX_IMAGE_BYTES is 10 MB; send 10 MB + 1
        big = b"x" * (10 * 1024 * 1024 + 1)
        response = client.post(
            "/analyze",
            files={"file": ("large.jpg", big, "image/jpeg")},
        )
    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()


def test_analyze_guardrail_rejection_returns_400(
    client: TestClient,
    valid_image_bytes: bytes,
    guardrail_check_not_food,
) -> None:
    """When pipeline raises GuardrailRejection, endpoint returns 400."""
    with patch("meal_analysis.api.main.run_analysis_pipeline", new_callable=AsyncMock) as m_run:
        m_run.side_effect = GuardrailRejection("Input guardrails failed: not food", guardrail_check_not_food)
        response = client.post(
            "/analyze",
            files={"file": ("meal.jpg", valid_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 400
    assert "not food" in response.json()["detail"]


def test_analyze_safety_rejection_returns_400(
    client: TestClient,
    valid_image_bytes: bytes,
    safety_checks_failed,
) -> None:
    """When pipeline raises SafetyRejection, endpoint returns 400."""
    with patch("meal_analysis.api.main.run_analysis_pipeline", new_callable=AsyncMock) as m_run:
        m_run.side_effect = SafetyRejection("Output safety check failed", safety_checks_failed)
        response = client.post(
            "/analyze",
            files={"file": ("meal.jpg", valid_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 400
    assert "safety" in response.json()["detail"].lower()


def test_analyze_agent_parse_error_returns_502(
    client: TestClient,
    valid_image_bytes: bytes,
) -> None:
    """When pipeline raises AgentParseError, endpoint returns 502."""
    with patch("meal_analysis.api.main.run_analysis_pipeline", new_callable=AsyncMock) as m_run:
        m_run.side_effect = AgentParseError(
            "meal_analysis",
            "invalid JSON",
            raw_content="not json",
        )
        response = client.post(
            "/analyze",
            files={"file": ("meal.jpg", valid_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 502
    assert "invalid output" in response.json()["detail"].lower()


def test_analyze_http_status_error_429_returns_429(
    client: TestClient,
    valid_image_bytes: bytes,
) -> None:
    """When upstream returns 429, endpoint returns 429."""
    with patch("meal_analysis.api.main.run_analysis_pipeline", new_callable=AsyncMock) as m_run:
        m_run.side_effect = httpx.HTTPStatusError(
            "429",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
        response = client.post(
            "/analyze",
            files={"file": ("meal.jpg", valid_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"].lower()


def test_analyze_http_status_error_5xx_returns_503(
    client: TestClient,
    valid_image_bytes: bytes,
) -> None:
    """When upstream returns 5xx, endpoint returns 503."""
    with patch("meal_analysis.api.main.run_analysis_pipeline", new_callable=AsyncMock) as m_run:
        m_run.side_effect = httpx.HTTPStatusError(
            "500",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        response = client.post(
            "/analyze",
            files={"file": ("meal.jpg", valid_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 503
    assert "Upstream" in response.json()["detail"]


def test_analyze_request_error_returns_503(
    client: TestClient,
    valid_image_bytes: bytes,
) -> None:
    """When upstream raises RequestError, endpoint returns 503."""
    with patch("meal_analysis.api.main.run_analysis_pipeline", new_callable=AsyncMock) as m_run:
        m_run.side_effect = httpx.RequestError("network error")
        response = client.post(
            "/analyze",
            files={"file": ("meal.jpg", valid_image_bytes, "image/jpeg")},
        )
    assert response.status_code == 503
    assert "request failed" in response.json()["detail"].lower()
