#!/usr/bin/env bash
set -euo pipefail

# Path to Turkish fastText vectors (.bin or .vec)
FASTTEXT_PATH="${FASTTEXT_PATH:-cc.tr.300.bin}"

# Where your fraction-specific vocab files live:
# Expected pattern per task: ${VOCAB_ROOT}/${task}/validation/top_${frac}.txt
VOCAB_ROOT="${VOCAB_ROOT:-vocab_files}"

# Output root directory for results
OUT_ROOT="${OUT_ROOT:-runs_lstm_word}"

# Tasks to run (extend if you want more)
TASKS=("mnli") 

# Fractions to sweep (must match your vocab filenames)
FRACS=("1")

# Hyperparameters (override via env if needed)
MAX_SEQ_LEN=${MAX_SEQ_LEN:-128}
BATCH_SIZE=${BATCH_SIZE:-128}
EPOCHS=${EPOCHS:-5}
LR=${LR:-2e-3}
HIDDEN=${HIDDEN:-256}
LAYERS=${LAYERS:-1}
DROPOUT=${DROPOUT:-0.3}
SEED=${SEED:-42}
CACHE_DIR="${CACHE_DIR:-}"

# Optional: set to 1 to freeze fastText embeddings
FREEZE="${FREEZE:-0}"

mkdir -p "$OUT_ROOT"

for task in "${TASKS[@]}"; do
  for frac in "${FRACS[@]}"; do
    vocab_file="${VOCAB_ROOT}/${task}/top_${frac}.txt"
    out_dir="${OUT_ROOT}/${task}/frac_${frac}"
    if [[ ! -f "$vocab_file" ]]; then
      echo "[WARN] Missing vocab file: $vocab_file — skipping."
      continue
    fi
    echo "Running task=${task} frac=${frac}"
    python3 trainer.py \
      --task_name "${task}" \
      --vocab_path "${vocab_file}" \
      --fasttext_path "${FASTTEXT_PATH}" \
      --max_seq_len ${MAX_SEQ_LEN} \
      --batch_size ${BATCH_SIZE} \
      --num_epochs ${EPOCHS} \
      --lr ${LR} \
      --output_dir "${out_dir}" \
      --hidden_size ${HIDDEN} \
      --num_layers ${LAYERS} \
      --dropout ${DROPOUT} \
      --seed ${SEED} \
      ${CACHE_DIR:+--cache_dir "$CACHE_DIR"} \
      $( [[ "$FREEZE" == "1" ]] && echo "--freeze_embeddings" )
  done
done
