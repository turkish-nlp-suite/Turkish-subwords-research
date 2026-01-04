#!/usr/bin/env bash


FRACS=("0.75" "0.8" "0.85" "0.9" "0.91" "0.92" "0.93" "0.94" "0.95" "0.96" "0.97" "0.98" "0.99", "1")

OUT_ROOT="${OUT_ROOT:-runs_ner_word}"
VOCAB_ROOT="${VOCAB_ROOT:-vocab_files}"

for frac in "${FRACS[@]}"; do
  vocab_file="${VOCAB_ROOT}/NER/top_${frac}.txt"
  out_dir="${OUT_ROOT}/frac_${frac}"

  python3 train_ner_word.py \
     --dataset turkish-nlp-suite/turkish-wikiNER \
     --epochs 10 --batch_size 256 --hidden 256 --layers 1 --dropout 0.3 \
     --vocab_file "${vocab_file}" \
     --use_fasttext \
     --fasttext_path ../../glue/training/cc.tr.300.bin \
     --output_dir "${out_dir}"
done
