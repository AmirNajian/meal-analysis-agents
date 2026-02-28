"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from meal_analysis.client import OpenAIClient


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


def main() -> None:
    """CLI entry (e.g. for `uv run python -m meal_analysis.api.main`)."""
    import uvicorn
    uvicorn.run("meal_analysis.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
