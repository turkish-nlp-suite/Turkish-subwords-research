#!/usr/bin/env bash
set -euo pipefail

TASKS=("POS")

# Parent directory for all per-task vocab dirs
PARENT_DIR="../training"

# Ensure parent exists

for TASK in "${TASKS[@]}"; do
  OUT_DIR="${PARENT_DIR}/vocab_files/${TASK}"
  mkdir -p "$OUT_DIR"
  echo "Writing vocabs for task=${TASK} into ${OUT_DIR}"
  python3 write_training_vocabs.py --task "$TASK" --out_dir "$OUT_DIR"
done
  


