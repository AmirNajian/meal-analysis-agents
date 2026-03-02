#!/usr/bin/env bash
# Run evals pipeline for a set of VLMs and collect metrics into a single JSON.
# Usage:
#   ./scripts/run_evals_pipeline.sh              # run pipeline + collect metrics (default models)
#   EVAL_MODELS="gpt-4o gpt-4o-mini" ./scripts/run_evals_pipeline.sh
#   ./scripts/run_evals_pipeline.sh --metrics-only   # only aggregate metrics from existing result files

set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

# Config: env or defaults
EVAL_DATA_DIR="${EVAL_DATA_DIR:-$REPO_ROOT/data}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-$EVAL_DATA_DIR/results}"
EVAL_OUTPUT_PREFIX="${EVAL_OUTPUT_PREFIX:-eval_results}"
EVAL_METRICS_FILE="${EVAL_METRICS_FILE:-$EVAL_RESULTS_DIR/eval_metrics_summary.json}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-10}"
# Default: 4 production VLMs (smaller set for speed). 
# Full set: gpt-5.2 gpt-5.2-pro gpt-5.1 gpt-5 gpt-5-mini gpt-4o gpt-4o-mini gpt-4.1 gpt-4.1-mini gpt-4-turbo
EVAL_MODELS="${EVAL_MODELS:-gpt-5.2 gpt-5.1 gpt-4o gpt-4o-mini}"

METRICS_ONLY=false
for arg in "$@"; do
  if [ "$arg" = "--metrics-only" ]; then
    METRICS_ONLY=true
    break
  fi
done

_ts() { date '+%H:%M:%S'; }

if [ "$METRICS_ONLY" = false ]; then
  MODEL_COUNT=$(echo $EVAL_MODELS | wc -w | tr -d ' ')
  echo "[$(_ts)] Running evals for $MODEL_COUNT model(s): $EVAL_MODELS"
  echo "[$(_ts)] Results -> $EVAL_RESULTS_DIR"
  mkdir -p "$EVAL_RESULTS_DIR"
  export PYTHONUNBUFFERED=1
  uv run python -m evals.runner \
    --data-dir "$EVAL_DATA_DIR" \
    --max-concurrency "$EVAL_MAX_CONCURRENCY" \
    --models $EVAL_MODELS \
    -o "${EVAL_RESULTS_DIR}/${EVAL_OUTPUT_PREFIX}.json"
  echo "[$(_ts)] Pipeline run finished."
fi

echo "[$(_ts)] Collecting metrics into $EVAL_METRICS_FILE"
export PYTHONUNBUFFERED=1
uv run python -m evals.collect_metrics \
  --output-prefix "$EVAL_OUTPUT_PREFIX" \
  --results-dir "$EVAL_RESULTS_DIR" \
  --data-dir "$EVAL_DATA_DIR" \
  --metrics-out "$EVAL_METRICS_FILE" \
  $EVAL_MODELS
