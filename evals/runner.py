"""Eval runner: discover image–JSON pairs, run pipeline per sample with asyncio."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx

from meal_analysis.agents.guardrail_check import AgentParseError
from meal_analysis.api.pipeline import GuardrailRejection, SafetyRejection, run_analysis_pipeline
from meal_analysis.client import OpenAIClient
from meal_analysis.config import get_config
from meal_analysis.schemas import EvalSample, EvalSampleResult, GroundTruthRecord


# Default data dir: project root / data (assume evals/ is at repo root)
def _default_data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def load_ground_truth(json_path: Path) -> GroundTruthRecord:
    """Load and parse a ground-truth JSON file into a GroundTruthRecord."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return GroundTruthRecord.model_validate(data)


async def run_one(
    sample: EvalSample,
    client: OpenAIClient,
    model: str,
) -> EvalSampleResult:
    """Run the pipeline for one sample; return a result with latency and success/error."""
    image_bytes = sample.image_path.read_bytes()
    sample_id = sample.sample_id
    start = time.perf_counter()
    try:
        response = await run_analysis_pipeline(
            image_bytes=image_bytes,
            client=client,
            model=model,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return EvalSampleResult(
            sample_id=sample_id,
            latency_ms=latency_ms,
            success=True,
            response=response,
        )
    except (
        GuardrailRejection,
        SafetyRejection,
        AgentParseError,
        httpx.HTTPStatusError,
        httpx.RequestError,
    ) as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return EvalSampleResult(
            sample_id=sample_id,
            latency_ms=latency_ms,
            success=False,
            error_class=type(e).__name__,
            error_message=str(e),
        )


def discover_pairs(
    *,
    images_dir: Path | None = None,
    json_dir: Path | None = None,
    data_dir: Path | None = None,
) -> list[EvalSample]:
    """Discover image–JSON pairs by basename (stem) match.

    - Lists *.jpeg and *.jpg under images_dir.
    - For each image, looks for {stem}.json under json_dir.
    - Only includes samples where both files exist.
    - Returns list sorted by sample_id for stable order.

    Parameters
    ----------
    images_dir
        Directory containing meal images. Default: data_dir / "images".
    json_dir
        Directory containing ground-truth JSONs. Default: data_dir / "json-files".
    data_dir
        Base data directory. Default: project root / "data".
        Ignored if both images_dir and json_dir are provided.
    """
    base = data_dir if data_dir is not None else _default_data_dir()
    img_dir = images_dir if images_dir is not None else base / "images"
    js_dir = json_dir if json_dir is not None else base / "json-files"

    pairs: list[EvalSample] = []
    seen_stems: set[str] = set()

    for ext in ("*.jpeg", "*.jpg"):
        for image_path in sorted(img_dir.glob(ext)):
            stem = image_path.stem
            if stem in seen_stems:
                continue
            json_path = js_dir / f"{stem}.json"
            if json_path.exists():
                pairs.append(EvalSample(image_path=image_path, json_path=json_path))
                seen_stems.add(stem)

    pairs.sort(key=lambda s: s.sample_id)
    return pairs
