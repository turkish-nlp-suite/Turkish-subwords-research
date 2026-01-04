#!/usr/bin/env bash


FRACS=("0.75" "0.8" "0.85" "0.9" "0.91" "0.92" "0.93" "0.94" "0.95" "0.96" "0.97" "0.98" "0.99", "1")

OUT_ROOT="${OUT_ROOT:-runs_pos_word}"
VOCAB_ROOT="${VOCAB_ROOT:-vocab_files}"

for frac in "${FRACS[@]}"; do
  vocab_file="${VOCAB_ROOT}/POS/top_${frac}.txt"
  out_dir="${OUT_ROOT}/frac_${frac}"

  python3 pos_dep_morph_word.py \
     --train tr_boun-ud-train.conllu \
     --dev tr_boun-ud-dev.conllu \
     --test tr_boun-ud-test.conllu \
     --epochs 20 --batch_size 128 \
     --vocab_file "${vocab_file}" \
     --output_dir "${out_dir}" \
     --use_fasttext \
     --fasttext_path ../../glue/training/cc.tr.300.bin \
     --output_dir "${out_dir}" \
     --fasttext_loader "official"
done

