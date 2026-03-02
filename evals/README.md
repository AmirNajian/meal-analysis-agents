# Evals

Run the meal-analysis pipeline (guardrail → meal inference → safety) on image–JSON pairs and write per-sample results for downstream metrics.

## Prerequisites

- `OPENAI_API_KEY` set (or in `.env`)
- Data layout: `data/images/*.jpeg` (or `.jpg`) and `data/json-files/*.json`, matched by filename stem

## Eval pipeline overview

The eval workflow has three steps:

1. **Run** — `evals.runner` runs the pipeline for each (image, ground-truth JSON) pair, optionally for multiple models; writes per-model result JSON(s).
2. **Collect metrics** — `evals.collect_metrics` reads those JSONs and ground truth, computes per-model metrics, and writes a single summary JSON.
3. **Render table** — `evals.render_results_table` reads the summary JSON and writes the Markdown results table (e.g. to `docs/RESULTS_TABLE.md`).

The script `scripts/run_evals_pipeline.sh` runs all three steps (or steps 2–3 only with `--metrics-only`).

## Using scripts/run_evals_pipeline.sh

Run from the **repo root**.

**Full run** (runner → collect_metrics → render_results_table):

```bash
./scripts/run_evals_pipeline.sh
```

**Metrics only** (aggregate from existing result files and re-render the table; no new API calls):

```bash
./scripts/run_evals_pipeline.sh --metrics-only
```

**Environment variables** (optional overrides):

| Variable | Default | Description |
|----------|---------|-------------|
| `EVAL_DATA_DIR` | repo root / `data` | Base data directory |
| `EVAL_RESULTS_DIR` | `EVAL_DATA_DIR/results` | Directory for per-model result files |
| `EVAL_OUTPUT_PREFIX` | `eval_results` | Stem of result files |
| `EVAL_METRICS_FILE` | `EVAL_RESULTS_DIR/eval_metrics_summary.json` | Output path for summary JSON |
| `EVAL_TABLE_FILE` | `repo_root/docs/RESULTS_TABLE.md` | Output path for Markdown table |
| `EVAL_MAX_CONCURRENCY` | `10` | Max concurrent pipeline runs |
| `EVAL_MODELS` | `gpt-5.2 gpt-5.1 gpt-4o gpt-4o-mini` | Space-separated model IDs |

Example with a subset of models:

```bash
EVAL_MODELS="gpt-4o gpt-4o-mini" ./scripts/run_evals_pipeline.sh
```

**Outputs:**

- Per-model result files: `<EVAL_RESULTS_DIR>/<EVAL_OUTPUT_PREFIX>_<model_slug>.json`
- Summary: `EVAL_METRICS_FILE`
- Table: `EVAL_TABLE_FILE`

## Metrics

Metrics are defined in [evals/metrics.py](evals/metrics.py). All percentage metrics are 0–100.

| Metric | Description |
|--------|-------------|
| **guardrails_pct** | % of samples (with response) where all guardrail booleans match ground truth. |
| **safety_pct** | % of samples (with response) where all safety booleans match ground truth. |
| **meal_pct** | Composite 0–100: recommendation (3-class: green/yellow/red), macros (APE-based score), ingredients (match by normalized name + impact). Text quality is stubbed. |
| **run_composite** | `0.2 * guardrails_pct + 0.5 * meal_pct + 0.3 * safety_pct` (0–100). |
| **p50_latency_ms** | Median end-to-end latency in ms (all samples). |
| **avg_input_tokens**, **avg_output_tokens** | Mean over results that have token data (otherwise omitted). |
| **p50_guardrail_latency_ms**, **p50_meal_latency_ms**, **p50_safety_latency_ms** | Per-agent median latencies when available. |

The summary JSON and results table are built from these metrics. `compute_metrics` and the functions in `evals/metrics.py` are the source of truth.

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

This writes `eval_results_gpt4o.json` and `eval_results_gpt4o-mini.json` and logs metrics for each. Alternatively, run twice with a fixed output path per model:

```bash
uv run python -m evals.runner --model gpt-4o -o eval_results_gpt4o.json
uv run python -m evals.runner --model gpt-4o-mini -o eval_results_gpt4o_mini.json
```

The CLI logs progress as each sample completes (e.g. `Completed 15/72...`), a final success count, and **metrics** (run composite, guardrails %, safety %, meal %, P50 latency ms) computed from the written results and ground truth. You can also re-compute from existing files via `compute_metrics_from_file` or the collect_metrics step.

## Output

**Per-model result JSON** (from the runner):

- **`results`**: list of per-sample objects (`sample_id`, `latency_ms`, `success`, `response` or `error_class`/`error_message`, optional `input_tokens`/`output_tokens`)
- **`meta`** (when using CLI): `model`, `max_concurrency`, `timestamp`

Token usage (`input_tokens`, `output_tokens`) may be `null` until the pipeline or client exposes it.

**Summary JSON** (from collect_metrics):

- **`timestamp`** — When the summary was generated
- **`output_prefix`**, **`data_dir`** — Config used
- **`models`** — Object keyed by model name; each value is the metrics dict (guardrails_pct, safety_pct, meal_pct, run_composite, p50_latency_ms, etc.)

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
- **`compute_metrics_from_file(results_path, data_dir=..., images_dir=..., json_dir=...)`** – load results, build ground truth from discovered pairs, return metrics dict

Metrics (see `evals/metrics.py`) are wired to runner output: the CLI computes and logs them after each run; you can also call `compute_metrics_from_file` on an existing results file to re-run metrics without re-running the pipeline.
