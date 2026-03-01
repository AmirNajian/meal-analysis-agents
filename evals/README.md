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
| `--output`, `-o` | `eval_results.json` | Output JSON path |

Example:

```bash
uv run python -m evals.runner --max-concurrency 2 -o my_results.json
```

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
- **`run_all(samples, max_concurrency=5)`** – run all samples with semaphore-limited concurrency (async)
- **`write_results(results, path, model=..., max_concurrency=...)`** – write JSON file

Results can be fed into a separate metrics step (guardrails, safety, meal composite, P50 latency) as in the assignment spec.
