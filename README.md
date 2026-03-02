# Meal Analysis Agents

An agentic meal-analysis pipeline with specialized guardrail, inference, and safety agents that collaborate to turn meal images into clinically aware, user-friendly nutrition guidance.

## Eval results

Eval metrics and a **results table** (Model, Run composite, Avg input/output tokens, P50 latency) are produced when you run the evals pipeline. The table is generated from the latest run.

- **Results table:** [docs/RESULTS_TABLE.md](docs/RESULTS_TABLE.md) (generated)
- **Summary JSON:** [data/results/eval_metrics_summary.json](data/results/eval_metrics_summary.json)

To regenerate (run evals for all configured models, then collect metrics and render the table):

```bash
./scripts/run_evals_pipeline.sh
```

To only re-aggregate metrics and re-render the table from existing result files (no new API calls):

```bash
./scripts/run_evals_pipeline.sh --metrics-only
```
