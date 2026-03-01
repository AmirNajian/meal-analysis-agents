# Evals

Run the meal-analysis pipeline (guardrail → meal inference → safety) on image–JSON pairs and write per-sample results for downstream metrics.

## Prerequisites

- `OPENAI_API_KEY` set (or in `.env`)
- Data layout: `data/images/*.jpeg` (or `.jpg`) and `data/json-files/*.json`, matched by filename stem

## CLI

From the project root:

```bash
uv run python -m evals.runner [OPTIONS]
```

| Option | Default | Description |
|--------|--------|-------------|
| `--data-dir` | project root / `data` | Base data directory |
| `--images-dir` | `data_dir/images` | Meal images directory |
| `--json-dir` | `data_dir/json-files` | Ground-truth JSON directory |
| `--max-concurrency` | `5` | Max concurrent pipeline runs |
| `--output`, `-o` | `eval_results.json` | Output JSON path; with `--models`, used as prefix for per-model files |
| `--model` | from config | Override model for a single run |
| `--models` | — | Run evals for each listed model; writes `<output_stem>_<model_slug>.json` per model |

Example (single run):

```bash
uv run python -m evals.runner --max-concurrency 2 -o my_results.json
```

Run evals for 2+ models (single invocation):

```bash
uv run python -m evals.runner --models gpt-4o gpt-4o-mini -o eval_results.json
```

This writes `eval_results_gpt4o.json` and `eval_results_gpt4o-mini.json` and prints metrics for each. Alternatively, run twice with a fixed output path per model:

```bash
uv run python -m evals.runner --model gpt-4o -o eval_results_gpt4o.json
uv run python -m evals.runner --model gpt-4o-mini -o eval_results_gpt4o_mini.json
```

The CLI prints progress as each sample completes (e.g. `Completed 15/72...`), a final success count, and **metrics** (run composite, guardrails %, safety %, meal %, P50 latency ms) computed from the written results and ground truth.

## Output

A single JSON file with:

- **`results`**: list of per-sample objects (`sample_id`, `latency_ms`, `success`, `response` or `error_class`/`error_message`, optional `input_tokens`/`output_tokens`)
- **`meta`** (when using CLI): `model`, `max_concurrency`, `timestamp`

Token usage (`input_tokens`, `output_tokens`) is not collected yet; fields are `null` until the pipeline or client exposes usage.

## Programmatic use

```python
from evals import discover_pairs, run_all, write_results
from pathlib import Path

samples = discover_pairs(data_dir=Path("data"))
results = asyncio.run(run_all(samples, max_concurrency=5))
write_results(results, Path("eval_results.json"), model="gpt-4o", max_concurrency=5)
```

- **`discover_pairs(images_dir=..., json_dir=..., data_dir=...)`** – find image–JSON pairs by stem
- **`load_ground_truth(json_path)`** – parse a ground-truth JSON into `GroundTruthRecord`
- **`run_one(sample, client, model)`** – run pipeline for one sample (async)
- **`run_all(samples, max_concurrency=5, on_progress=None)`** – run all samples with semaphore-limited concurrency (async); optional `on_progress(completed, total)` for progress output
- **`write_results(results, path, model=..., max_concurrency=...)`** – write JSON file
- **`load_results(path)`** – load a results JSON file back into `list[EvalSampleResult]`
- **`compute_metrics_from_file(results_path, data_dir=..., images_dir=..., json_dir=...)`** – load results, build ground truth from discovered pairs, return metrics dict (`guardrails_pct`, `safety_pct`, `meal_pct`, `run_composite`, `p50_latency_ms`)

Metrics (see `evals/metrics.py`) are wired to runner output: the CLI computes and prints them after each run; you can also call `compute_metrics_from_file` on an existing results file to re-run metrics without re-running the pipeline.
