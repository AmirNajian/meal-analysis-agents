# Meal Analysis Agents

An agentic meal-analysis pipeline with specialized guardrail, inference, and safety agents that collaborate to turn meal images into clinically aware, user-friendly nutrition guidance.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for install and run
- Optional: `OPENAI_API_KEY` in environment or `.env` (required for the API and evals)

## Setup

From the repo root:

```bash
uv sync
```

## Running the FastAPI server locally

From the repo root:

```bash
uv run python -m meal_analysis.api.main
```

The server runs at `http://0.0.0.0:8000` with reload. Use `POST /analyze` to upload a meal image (e.g. JPEG) and receive the pipeline result. Interactive API docs: `http://localhost:8000/docs`.

## Testing

From the repo root:

```bash
uv run pytest tests/ -v
```

Test coverage with a terminal report and list of missing lines:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

## Module and repo structure

| Path | Description |
|------|-------------|
| **`src/meal_analysis/`** | Main package: API ([`api/main.py`](src/meal_analysis/api/main.py), pipeline), agents (guardrail, meal, safety), client, config, schemas. |
| **`evals/`** | Eval runner, metrics, collect_metrics, render_results_table; runs the same pipeline on image–JSON pairs from `data/`. |
| **`data/`** | Curated dataset: `images/` and `json-files/` (ground truth). See [data/README.md](data/README.md). |
| **`docs/`** | Project docs and generated artifacts (e.g. results table). See [docs/README.md](docs/README.md). |
| **`scripts/`** | Shell script to run the full evals pipeline and render the table (see Evals below). |
| **`tests/`** | Unit tests (e.g. `tests/unit/`). |

## Evals and results

Eval metrics and a **results table** (Model, Run composite, Avg input/output tokens, P50 latency, etc.) are produced when you run the evals pipeline.

- **Results table:** [docs/RESULTS_TABLE.md](docs/RESULTS_TABLE.md) (generated)
- **Summary JSON:** [data/results/eval_metrics_summary.json](data/results/eval_metrics_summary.json)

**Full pipeline** (run evals for all configured models, then collect metrics and render the table):

```bash
./scripts/run_evals_pipeline.sh
```

**Metrics only** (re-aggregate from existing result files and re-render the table; no new API calls):

```bash
./scripts/run_evals_pipeline.sh --metrics-only
```

For full pipeline details, metrics definitions, and script options, see [evals/README.md](evals/README.md).

## More documentation

- **[docs/README.md](docs/README.md)** — What lives in `docs/`: project documentation and generated artifacts; RESULTS_TABLE.md and how it’s produced.
- **[data/README.md](data/README.md)** — Dataset layout (`images/`, `json-files/`), pairing by basename, ground-truth JSON structure (guardrailCheck, safetyChecks, mealAnalysis), and how evals compare against these fields.
- **[evals/README.md](evals/README.md)** — Evals in depth: CLI for the runner, pipeline stages, output format, programmatic API, and full metrics and pipeline script usage.
