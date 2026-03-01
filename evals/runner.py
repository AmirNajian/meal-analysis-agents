"""Eval runner: discover image–JSON pairs, run pipeline per sample with asyncio.

Token usage (input_tokens/output_tokens) is not collected in this version:
EvalSampleResult fields remain None until run_analysis_pipeline or a client
wrapper exposes per-call usage. See step 5 in the eval runner plan.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Callable

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
    """Run the pipeline for one sample; return a result with latency and success/error.

    Token usage (input_tokens, output_tokens) is not populated; the pipeline
    does not return it. To add later: extend run_analysis_pipeline to aggregate
    and return usage, or use a client wrapper that records it.
    """
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


async def run_all(
    samples: list[EvalSample],
    *,
    max_concurrency: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[EvalSampleResult]:
    """Run the pipeline for all samples with a semaphore-limited concurrency.

    Uses a single shared OpenAIClient (created and closed inside this function).
    Any unexpected exception from run_one is turned into an EvalSampleResult
    with success=False so the returned list has one result per sample.

    If on_progress(completed, total) is given, it is called after each sample
    completes (e.g. for CLI progress output).
    """
    model = get_config().openai_model
    semaphore = asyncio.Semaphore(max_concurrency)
    total = len(samples)
    progress_lock = asyncio.Lock()
    completed: list[int] = [0]

    async with OpenAIClient() as client:
        async def run_with_sem(sample: EvalSample) -> EvalSampleResult | BaseException:
            async with semaphore:
                try:
                    return await run_one(sample, client, model)
                finally:
                    if on_progress is not None:
                        async with progress_lock:
                            completed[0] += 1
                            on_progress(completed[0], total)

        raw = await asyncio.gather(
            *[run_with_sem(s) for s in samples],
            return_exceptions=True,
        )

    results: list[EvalSampleResult] = []
    for i, r in enumerate(raw):
        if isinstance(r, EvalSampleResult):
            results.append(r)
        else:
            exc = r if isinstance(r, BaseException) else Exception(r)
            results.append(
                EvalSampleResult(
                    sample_id=samples[i].sample_id,
                    latency_ms=0.0,
                    success=False,
                    error_class=type(exc).__name__,
                    error_message=str(exc),
                )
            )
    return results


def write_results(
    results: list[EvalSampleResult],
    path: Path,
    *,
    model: str | None = None,
    max_concurrency: int | None = None,
) -> None:
    """Write eval results to a JSON file (list of result dicts plus optional metadata)."""
    payload: dict = {
        "results": [r.model_dump(mode="json") for r in results],
    }
    if model is not None or max_concurrency is not None:
        payload["meta"] = {}
        if model is not None:
            payload["meta"]["model"] = model
        if max_concurrency is not None:
            payload["meta"]["max_concurrency"] = max_concurrency
        payload["meta"]["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run meal-analysis pipeline on image–JSON pairs and write results.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Base data directory (default: project root / data)",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory containing meal images (default: data_dir / images)",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=None,
        help="Directory containing ground-truth JSONs (default: data_dir / json-files)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        metavar="N",
        help="Max concurrent pipeline runs (default: 5)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("eval_results.json"),
        metavar="FILE",
        help="Output JSON file path (default: eval_results.json)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry: discover pairs, run_all, write results."""
    args = _parse_args()
    samples = discover_pairs(
        data_dir=args.data_dir,
        images_dir=args.images_dir,
        json_dir=args.json_dir,
    )
    if not samples:
        print("No image–JSON pairs found.")
        return
    print(f"Running pipeline on {len(samples)} samples (max_concurrency={args.max_concurrency})...")
    results = asyncio.run(
        run_all(
            samples,
            max_concurrency=args.max_concurrency,
            on_progress=lambda n, total: print(f"  Completed {n}/{total}..."),
        ),
    )
    write_results(
        results,
        args.output,
        model=get_config().openai_model,
        max_concurrency=args.max_concurrency,
    )
    print(f"Wrote {len(results)} results to {args.output}")
    success_count = sum(1 for r in results if r.success)
    print(f"Success: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
