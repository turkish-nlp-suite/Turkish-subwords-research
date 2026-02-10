#!/usr/bin/env bash
set -u

log_fail() {
  echo "[FAIL] $*" >&2
}

for vocab in 2 5 10 20 32 52 128; do
  for ds in minimal books alldata; do
    model_name="turkish-nlp-suite/bert-${vocab}K-${ds}"
    tokenizer_name="turkish-nlp-suite/wordpiece_${vocab}k_cased_${ds}"
    task="sst2"

    outdir="outputs/${model_name}/${task}"
    mkdir -p "${outdir}"

    python3 run.py \
      --model_name_or_path "${model_name}" \
      --tokenizer_name "${tokenizer_name}" \
      --task_name "${task}" \
      --max_seq_length 128 \
      --output_dir "${outdir}" \
      --num_train_epochs 3 \
      --learning_rate 3e-5 \
      --per_device_train_batch_size 128 \
      --per_device_eval_batch_size 128 \
      --seed 42 \
      --do_train \
      --do_eval \
      --save_strategy epoch \
      --save_total_limit 1
  done
done
