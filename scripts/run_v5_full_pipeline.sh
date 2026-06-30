#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${1:?usage: run_v5_full_pipeline.sh LOG_PATH}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$(dirname "$LOG_PATH")"
exec >>"$LOG_PATH" 2>&1

cd "$REPO_DIR"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

caffeinate -dimsu -w "$$" &
CAFFEINATE_PID=$!
trap 'kill "$CAFFEINATE_PID" 2>/dev/null || true' EXIT

echo "START v5 full pipeline $(date)"
echo "WORKDIR $(pwd)"
echo "SHELL_PID $$"
echo "CAFFEINATE_PID $CAFFEINATE_PID"

echo "CALIBRATE $(date)"
python -u -m qrc_dataset_profiler.run_v5_protocol calibrate \
  --out results_calibration_v5 \
  --sweep-n-per-template 20 \
  --rows-per-family 20 \
  --fast \
  --seeds 1

echo "DISCOVERY $(date)"
python -u -m qrc_dataset_profiler.run_v5_protocol evaluate-selection \
  --selection results_frontier_v4_selection/frontier_evaluation_selection.csv \
  --calibration-config results_calibration_v5/frozen_v5_config.json \
  --out results_frontier_v5_discovery \
  --split discovery \
  --fast \
  --seeds 1 \
  --include-nvar \
  --checkpoint-every 100

echo "SUMMARIZE DISCOVERY $(date)"
python -u -m qrc_dataset_profiler.run_v5_protocol summarize \
  --evaluated-table results_frontier_v5_discovery/frontier_discovery_evaluated_v5_multi_qrc.csv \
  --out results_frontier_v5_discovery/v5_evaluation_summary.csv

echo "VALIDATION $(date)"
python -u -m qrc_dataset_profiler.run_v5_protocol evaluate-selection \
  --selection results_frontier_v4_selection/frontier_evaluation_selection.csv \
  --calibration-config results_calibration_v5/frozen_v5_config.json \
  --out results_frontier_v5_validation \
  --split validation \
  --fast \
  --seeds 1 \
  --include-nvar \
  --checkpoint-every 100

echo "SUMMARIZE VALIDATION $(date)"
python -u -m qrc_dataset_profiler.run_v5_protocol summarize \
  --evaluated-table results_frontier_v5_validation/frontier_validation_evaluated_v5_multi_qrc.csv \
  --out results_frontier_v5_validation/v5_evaluation_summary.csv

echo "DONE v5 full pipeline $(date)"
