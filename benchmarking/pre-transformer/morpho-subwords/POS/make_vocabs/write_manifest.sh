#!/usr/bin/env bash
set -euo pipefail

TASKS=("POS")
PARENT_DIR="../training"
for TASK in "${TASKS[@]}"; do
  OUT_DIR="${PARENT_DIR}/vocab_files/${TASK}"
  mkdir -p "$OUT_DIR"
  echo "Writing subword vocabs for task=${TASK} into ${OUT_DIR}"
  python3 write_training_vocabs.py \
    --task "$TASK" \
    --out_dir "$OUT_DIR" \
    --coverage_targets_csv "${TASK}/coverage_subw_train_targets.csv" \
    --train_sub_counts "${TASK}/train_subwrd_counts.txt" \
    --prefix "top_"
done
