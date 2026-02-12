#!/usr/bin/env bash
set -euo pipefail

# Simple config
BASE_DIR="."
SPLIT="test"                 # e.g., validation|test|dev
TASKS="cola mrpc stsb mnli"
TARGETS="0.75,0.80,0.85,0.90,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1.00"

echo "Split: $SPLIT"
echo "Tasks: $TASKS"
echo "Targets: $TARGETS"

for TASK in $TASKS; do
  echo "==> Task: $TASK"
  python3 coverage_runner.py \
    --task "$TASK" \
    --base_dir "$BASE_DIR" \
    --split "$SPLIT" \
    --target_coverages "$TARGETS"
done
