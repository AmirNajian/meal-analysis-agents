"""FastAPI application entrypoint."""

from fastapi import FastAPI

app = FastAPI(
    title="meal-analysis-agents",
    description="Agentic meal-analysis pipeline with guardrail, inference, and safety agents.",
)


def main() -> None:
    """CLI entry (e.g. for `uv run python -m meal_analysis.api.main`)."""
    import uvicorn
    uvicorn.run("meal_analysis.api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
