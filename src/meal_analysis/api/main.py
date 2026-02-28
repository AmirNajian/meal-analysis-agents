"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile

from meal_analysis.client import OpenAIClient
from meal_analysis.config import get_config
from meal_analysis.schemas import AnalysisResponse
from meal_analysis.api.pipeline import (
    GuardrailRejection,
    SafetyRejection,
    run_analysis_pipeline,
)
from meal_analysis.agents.guardrail_check import AgentParseError

MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create OpenAIClient at startup, close it at shutdown."""
    client = OpenAIClient()
    app.state.openai_client = client
    yield
    await client.aclose()


app = FastAPI(
    title="meal-analysis-agents",
    description="Agentic meal-analysis pipeline with guardrail, inference, and safety agents.",
    lifespan=lifespan,
)


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(..., description="Meal image (e.g. JPEG)"),
) -> AnalysisResponse:
    """Run guardrail → meal analysis → safety checks; return combined result or 4xx/5xx."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing or invalid image file")

    image_bytes = await file.read()
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large (max {MAX_IMAGE_BYTES // (1024*1024)} MB)",
        )

    client: OpenAIClient = request.app.state.openai_client
    model = get_config().openai_model

    try:
        return await run_analysis_pipeline(
            image_bytes=image_bytes,
            client=client,
            model=model,
        )
    except GuardrailRejection as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except SafetyRejection as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except AgentParseError as e:
        raise HTTPException(
            status_code=502,
            detail="Model returned invalid output",
        ) from e
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
            ) from e
        raise HTTPException(
            status_code=503,
            detail="Upstream service error",
        ) from e
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail="Upstream request failed",
        ) from e


def main() -> None:
    """CLI entry (e.g. for `uv run python -m meal_analysis.api.main`)."""
    import uvicorn
    uvicorn.run("meal_analysis.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
