#!/usr/bin/env bash
set -euo pipefail

# Fractions you have available
FRACS=("0.75" "0.8" "0.85" "0.9")

OUT_ROOT="${OUT_ROOT:-runs_ner_subw}"
VOCAB_ROOT="${VOCAB_ROOT:-vocab_files/NER}"

DATASET="${DATASET:-turkish-nlp-suite/turkish-wikiNER}"
FT_PATH="${FT_PATH:-/home/daltinok_paloaltonetworks_com/duygu/subwords/word/glue/training/cc.tr.300.bin}"

EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN="${HIDDEN:-256}"
LAYERS="${LAYERS:-1}"
DROPOUT="${DROPOUT:-0.3}"
LR="${LR:-5e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"

for frac in "${FRACS[@]}"; do
  vocab_file="${VOCAB_ROOT}/top_${frac}.txt"
  out_dir="${OUT_ROOT}/frac_${frac}"
  mkdir -p "${out_dir}"

  echo "[Run] frac=${frac} vocab=${vocab_file} -> ${out_dir}"

  python3 train_ner_word.py \
     --dataset "${DATASET}" \
     --epochs "${EPOCHS}" \
     --batch_size "${BATCH_SIZE}" \
     --hidden "${HIDDEN}" \
     --layers "${LAYERS}" \
     --dropout "${DROPOUT}" \
     --lr "${LR}" \
     --weight_decay "${WEIGHT_DECAY}" \
     --vocab_file "${vocab_file}" \
     --use_subwords \
     --use_fasttext \
     --fasttext_path "${FT_PATH}" \
     --output_dir "${out_dir}"
done
