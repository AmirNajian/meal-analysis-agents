"""Unit tests for evals.runner (discover_pairs, run_one, run_all, write_results)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from evals.runner import (
    compute_metrics_from_file,
    discover_pairs,
    load_ground_truth,
    load_results,
    run_all,
    run_one,
    write_results,
)
from meal_analysis.schemas import EvalSample, EvalSampleResult


def test_discover_pairs_finds_matching_pairs(evals_fixture_dir: Path) -> None:
    """discover_pairs returns one pair per image stem that has a matching JSON; sorted by sample_id."""
    pairs = discover_pairs(
        images_dir=evals_fixture_dir / "images",
        json_dir=evals_fixture_dir / "json-files",
    )
    assert len(pairs) == 2
    sample_ids = [s.sample_id for s in pairs]
    assert sample_ids == ["meal_a", "meal_b"]
    for s in pairs:
        assert s.image_path.exists()
        assert s.json_path.exists()
        assert s.image_path.stem == s.json_path.stem


def test_discover_pairs_dedupes_by_stem(evals_fixture_dir: Path) -> None:
    """When the same stem appears as both .jpeg and .jpg, only one pair is returned."""
    pairs = discover_pairs(
        images_dir=evals_fixture_dir / "images",
        json_dir=evals_fixture_dir / "json-files",
    )
    meal_a_pairs = [p for p in pairs if p.sample_id == "meal_a"]
    assert len(meal_a_pairs) == 1


def test_discover_pairs_ignores_image_without_json(tmp_path: Path) -> None:
    """Images with no matching JSON file are not included."""
    (tmp_path / "images").mkdir()
    (tmp_path / "json-files").mkdir()
    (tmp_path / "images" / "no_pair.jpeg").write_bytes(b"\xff\xd8\xff")
    pairs = discover_pairs(
        images_dir=tmp_path / "images",
        json_dir=tmp_path / "json-files",
    )
    assert len(pairs) == 0


def test_discover_pairs_sorted_by_sample_id(evals_fixture_dir: Path) -> None:
    """Results are sorted by sample_id for stable order."""
    pairs = discover_pairs(
        images_dir=evals_fixture_dir / "images",
        json_dir=evals_fixture_dir / "json-files",
    )
    assert pairs == sorted(pairs, key=lambda s: s.sample_id)


# ---- load_ground_truth ----

def test_load_ground_truth(evals_fixture_dir: Path) -> None:
    """load_ground_truth parses fixture JSON into GroundTruthRecord."""
    gt = load_ground_truth(evals_fixture_dir / "json-files" / "meal_a.json")
    assert gt.title == "Fixture Meal"
    assert "meal_a" in gt.fileName
    assert gt.guardrailCheck.is_food is True
    assert gt.mealAnalysis.recommendation == "green"
    assert len(gt.mealAnalysis.ingredients) == 1
    assert gt.mealAnalysis.ingredients[0].name == "Lettuce"


# ---- run_one (mocked pipeline) ----


@pytest.mark.asyncio
async def test_run_one_success(
    evals_fixture_dir: Path,
    guardrail_check_passed,
    meal_analysis_result,
    safety_checks_passed,
) -> None:
    """run_one returns EvalSampleResult with success=True and response when pipeline succeeds."""
    from meal_analysis.schemas import AnalysisResponse

    sample = _run_one_sample(evals_fixture_dir)
    mock_client = MagicMock()
    expected = AnalysisResponse(
        guardrailCheck=guardrail_check_passed,
        mealAnalysis=meal_analysis_result,
        safetyChecks=safety_checks_passed,
    )
    with patch("evals.runner.run_analysis_pipeline", new_callable=AsyncMock, return_value=expected):
        result = await run_one(sample, mock_client, "gpt-4o")
    assert result.sample_id == "meal_a"
    assert result.success is True
    assert result.response is not None
    assert result.response.mealAnalysis.meal_title == meal_analysis_result.meal_title
    assert result.latency_ms >= 0
    assert result.error_class is None
    assert result.error_message is None


def _run_one_sample(evals_fixture_dir: Path) -> EvalSample:
    """Shared sample for run_one tests."""
    return EvalSample(
        image_path=evals_fixture_dir / "images" / "meal_a.jpeg",
        json_path=evals_fixture_dir / "json-files" / "meal_a.json",
    )


@pytest.mark.asyncio
async def test_run_one_guardrail_rejection(evals_fixture_dir: Path, guardrail_check_not_food) -> None:
    """run_one returns EvalSampleResult with success=False when pipeline raises GuardrailRejection."""
    from meal_analysis.api.pipeline import GuardrailRejection

    sample = _run_one_sample(evals_fixture_dir)
    mock_client = MagicMock()
    exc = GuardrailRejection("Input guardrails failed: not food", guardrail_check_not_food)
    with patch("evals.runner.run_analysis_pipeline", new_callable=AsyncMock, side_effect=exc):
        result = await run_one(sample, mock_client, "gpt-4o")
    assert result.sample_id == "meal_a"
    assert result.success is False
    assert result.response is None
    assert result.error_class == "GuardrailRejection"
    assert "not food" in (result.error_message or "")
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_run_one_safety_rejection(evals_fixture_dir: Path, safety_checks_failed) -> None:
    """run_one returns EvalSampleResult with success=False when pipeline raises SafetyRejection."""
    from meal_analysis.api.pipeline import SafetyRejection

    sample = _run_one_sample(evals_fixture_dir)
    mock_client = MagicMock()
    exc = SafetyRejection("Output safety check failed", safety_checks_failed)
    with patch("evals.runner.run_analysis_pipeline", new_callable=AsyncMock, side_effect=exc):
        result = await run_one(sample, mock_client, "gpt-4o")
    assert result.success is False
    assert result.error_class == "SafetyRejection"
    assert result.response is None
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_run_one_agent_parse_error(evals_fixture_dir: Path) -> None:
    """run_one returns EvalSampleResult with success=False when pipeline raises AgentParseError."""
    from meal_analysis.agents.guardrail_check import AgentParseError

    sample = _run_one_sample(evals_fixture_dir)
    mock_client = MagicMock()
    exc = AgentParseError("meal_analysis", "invalid JSON", "raw content")
    with patch("evals.runner.run_analysis_pipeline", new_callable=AsyncMock, side_effect=exc):
        result = await run_one(sample, mock_client, "gpt-4o")
    assert result.success is False
    assert result.error_class == "AgentParseError"
    assert "invalid JSON" in (result.error_message or "")
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_run_one_http_status_error(evals_fixture_dir: Path) -> None:
    """run_one returns EvalSampleResult with success=False when pipeline raises httpx.HTTPStatusError."""
    import httpx

    sample = _run_one_sample(evals_fixture_dir)
    mock_client = MagicMock()
    exc = httpx.HTTPStatusError("429", request=MagicMock(), response=MagicMock(status_code=429))
    with patch("evals.runner.run_analysis_pipeline", new_callable=AsyncMock, side_effect=exc):
        result = await run_one(sample, mock_client, "gpt-4o")
    assert result.success is False
    assert result.error_class == "HTTPStatusError"
    assert result.latency_ms >= 0


@pytest.mark.asyncio
async def test_run_one_request_error(evals_fixture_dir: Path) -> None:
    """run_one returns EvalSampleResult with success=False when pipeline raises httpx.RequestError."""
    import httpx

    sample = _run_one_sample(evals_fixture_dir)
    mock_client = MagicMock()
    with patch("evals.runner.run_analysis_pipeline", new_callable=AsyncMock, side_effect=httpx.RequestError("network error")):
        result = await run_one(sample, mock_client, "gpt-4o")
    assert result.success is False
    assert result.error_class == "RequestError"
    assert result.latency_ms >= 0


# ---- run_all ----

@pytest.mark.asyncio
async def test_run_all_returns_one_result_per_sample(evals_fixture_dir: Path) -> None:
    """run_all runs each sample and returns a result per sample (mocked run_one)."""
    samples = [
        EvalSample(image_path=evals_fixture_dir / "images" / "meal_a.jpeg", json_path=evals_fixture_dir / "json-files" / "meal_a.json"),
        EvalSample(image_path=evals_fixture_dir / "images" / "meal_b.jpg", json_path=evals_fixture_dir / "json-files" / "meal_b.json"),
    ]
    results_by_id = {
        "meal_a": EvalSampleResult(sample_id="meal_a", latency_ms=100.0, success=True),
        "meal_b": EvalSampleResult(sample_id="meal_b", latency_ms=200.0, success=False, error_class="GuardrailRejection", error_message="not food"),
    }

    async def mock_run_one(sample: EvalSample, client: object, model: str) -> EvalSampleResult:
        return results_by_id[sample.sample_id]

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = MagicMock()
    mock_cm.__aexit__.return_value = None
    with patch("evals.runner.OpenAIClient", return_value=mock_cm):
        with patch("evals.runner.run_one", side_effect=mock_run_one):
            got = await run_all(samples, max_concurrency=2)
    assert len(got) == 2
    assert got[0].sample_id == "meal_a" and got[0].success is True
    assert got[1].sample_id == "meal_b" and got[1].success is False


@pytest.mark.asyncio
async def test_run_all_uses_passed_model(evals_fixture_dir: Path) -> None:
    """run_all(..., model=X) passes model X to run_one."""
    samples = [
        EvalSample(image_path=evals_fixture_dir / "images" / "meal_a.jpeg", json_path=evals_fixture_dir / "json-files" / "meal_a.json"),
    ]
    seen_models: list[str] = []

    async def capture_model(sample: EvalSample, client: object, model: str) -> EvalSampleResult:
        seen_models.append(model)
        return EvalSampleResult(sample_id=sample.sample_id, latency_ms=1.0, success=True)

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = MagicMock()
    mock_cm.__aexit__.return_value = None
    with patch("evals.runner.OpenAIClient", return_value=mock_cm):
        with patch("evals.runner.run_one", side_effect=capture_model):
            await run_all(samples, max_concurrency=1, model="gpt-4o-mini")
    assert seen_models == ["gpt-4o-mini"]


@pytest.mark.asyncio
async def test_run_all_converts_gather_exception_to_result(evals_fixture_dir: Path) -> None:
    """When run_one raises an unexpected exception, run_all turns it into an EvalSampleResult."""
    samples = [
        EvalSample(image_path=evals_fixture_dir / "images" / "meal_a.jpeg", json_path=evals_fixture_dir / "json-files" / "meal_a.json"),
    ]

    async def mock_run_one_raises(*args: object, **kwargs: object) -> None:
        raise ValueError("unexpected")

    mock_cm = AsyncMock()
    mock_cm.__aenter__.return_value = MagicMock()
    mock_cm.__aexit__.return_value = None
    with patch("evals.runner.OpenAIClient", return_value=mock_cm):
        with patch("evals.runner.run_one", side_effect=mock_run_one_raises):
            got = await run_all(samples, max_concurrency=1)
    assert len(got) == 1
    assert got[0].sample_id == "meal_a"
    assert got[0].success is False
    assert got[0].error_class == "ValueError"
    assert "unexpected" in (got[0].error_message or "")


# ---- write_results ----

def test_write_results_writes_valid_json(tmp_path: Path) -> None:
    """write_results writes a JSON file with results and optional meta."""
    results = [
        EvalSampleResult(sample_id="a", latency_ms=50.0, success=True),
        EvalSampleResult(sample_id="b", latency_ms=100.0, success=False, error_class="GuardrailRejection", error_message="not food"),
    ]
    out = tmp_path / "out.json"
    write_results(results, out, model="gpt-4o", max_concurrency=2)
    data = json.loads(out.read_text())
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["sample_id"] == "a" and data["results"][0]["success"] is True
    assert data["results"][1]["sample_id"] == "b" and data["results"][1]["error_class"] == "GuardrailRejection"
    assert data["meta"]["model"] == "gpt-4o"
    assert data["meta"]["max_concurrency"] == 2
    assert "timestamp" in data["meta"]


def test_load_results_roundtrip(tmp_path: Path) -> None:
    """load_results parses a file written by write_results."""
    results = [
        EvalSampleResult(sample_id="a", latency_ms=50.0, success=True),
        EvalSampleResult(sample_id="b", latency_ms=100.0, success=False, error_class="X", error_message="y"),
    ]
    out = tmp_path / "r.json"
    write_results(results, out)
    loaded = load_results(out)
    assert len(loaded) == 2
    assert loaded[0].sample_id == "a" and loaded[0].latency_ms == 50.0 and loaded[0].success is True
    assert loaded[1].sample_id == "b" and loaded[1].success is False and loaded[1].error_class == "X"


def test_compute_metrics_from_file(evals_fixture_dir: Path) -> None:
    """compute_metrics_from_file loads results + ground truth and returns metrics."""
    from meal_analysis.schemas import AnalysisResponse

    # Use fixture ground truth so response matches exactly (100% scores)
    gt_a = load_ground_truth(evals_fixture_dir / "json-files" / "meal_a.json")
    resp = AnalysisResponse(
        guardrailCheck=gt_a.guardrailCheck,
        mealAnalysis=gt_a.mealAnalysis,
        safetyChecks=gt_a.safetyChecks,
    )
    results = [
        EvalSampleResult(sample_id="meal_a", latency_ms=10.0, success=True, response=resp),
        EvalSampleResult(sample_id="meal_b", latency_ms=20.0, success=True, response=resp),
    ]
    out = evals_fixture_dir / "metrics_test_results.json"
    write_results(results, out)
    metrics = compute_metrics_from_file(
        out,
        images_dir=evals_fixture_dir / "images",
        json_dir=evals_fixture_dir / "json-files",
    )
    assert metrics["run_composite"] == 100.0
    assert metrics["guardrails_pct"] == 100.0
    assert metrics["safety_pct"] == 100.0
    assert metrics["p50_latency_ms"] == 15.0
