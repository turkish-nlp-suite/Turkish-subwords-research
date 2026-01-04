#!/usr/bin/env bash


FRACS=("0.75" "0.8" "0.85" "0.85") 

OUT_ROOT="${OUT_ROOT:-runs_pos_subword}"
VOCAB_ROOT="${VOCAB_ROOT:-vocab_files}"

for frac in "${FRACS[@]}"; do
  vocab_file="${VOCAB_ROOT}/POS/top_${frac}.txt"
  out_dir="${OUT_ROOT}/frac_${frac}"

  python3 train.py \
     --train tr_boun-ud-train.conllu \
     --dev tr_boun-ud-dev.conllu \
     --test tr_boun-ud-test.conllu \
     --epochs 20 --batch_size 128 \
     --vocab_file "${vocab_file}" \
     --output_dir "${out_dir}" \
     --use_fasttext \
     --use_subwords \
     --fasttext_path /home/daltinok_paloaltonetworks_com/duygu/subwords/word/glue/training/cc.tr.300.bin \
     --subword_cache_train vocab_files/wcache_train.json \
     --subword_cache_test vocab_files/wcache_test.json \
     --subword_cache_dev vocab_files/wcache_dev.json \
     --output_dir "${out_dir}" \
     --fasttext_loader "official"
done
