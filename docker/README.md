# Docker

Run the meal-analysis API (and optional Gradio UI) with Docker so you don’t need Python/uv installed locally.

## Prerequisites

- Docker and Docker Compose
- A `.env` file in the **project root** with `OPENAI_API_KEY=...` (copy from `.env.example`; do not commit real keys)

## Quick start

From the **project root** (one level up from this folder):

```bash
docker compose up
```

This starts the API at `http://localhost:8000`. Optional: start the Gradio service as well (see `docker-compose.yml`) and open the UI (typically a second port).

## What’s included

- **`Dockerfile`** (project root): Python 3.12, uv, install deps, default CMD runs the FastAPI app.
- **`docker-compose.yml`** (project root):
  - **api**: Builds the image, runs the API; env loaded from `.env`.
  - **gradio** (optional): Same image, runs the Gradio app; uses `API_URL=http://api:8000` to call the API.

## Notes

- Evals are not run inside Docker by default; run them locally with `uv sync --extra dev` and `uv run python -m evals.runner` (or equivalent).
- Primary way to run the app is still **uv** (see main README); Docker is an alternative for reviewers or one-command demo.
