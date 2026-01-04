#!/usr/bin/env bash
set -u  # keep undefined-vars safety; do NOT use -e
# Do not set -o pipefail if you want to ignore failures; it’s harmless here but not needed.

log_fail() {
  echo "[FAIL] $*" >&2
}

for vocab in 2 5 10 20 32 52 128; do
  for ds in minimal books alldata; do
    model_name="turkish-nlp-suite/bert-${vocab}K-${ds}"
    tokenizer_name="turkish-nlp-suite/wordpiece_${vocab}k_cased_${ds}"

    for task in cola mnli mrpc sst-2 stsb; do
      # Set LR/BS
      if [[ "$task" == "cola" || "$task" == "sst-2" || "$task" == "mnli" ]]; then
        LR=3e-5; BS=128
      else
        LR=2e-5; BS=16
      fi

      outdir="outputs/${model_name}/${task}"
      mkdir -p "$outdir"

      echo ">>> Running task=${task} model=${model_name} tokenizer=${tokenizer_name} LR=${LR} BS=${BS}"

      python3 run.py \
        --model_name_or_path "${model_name}" \
        --tokenizer_name "${tokenizer_name}" \
        --task_name "${task}" \
        --max_seq_length 128 \
        --output_dir "${outdir}" \
        --num_train_epochs 3 \
        --learning_rate "${LR}" \
        --per_device_train_batch_size "${BS}" \
        --per_device_eval_batch_size "${BS}" \
        --seed 42 \
        --do_train \
        --do_eval \
        --save_strategy epoch \
        --save_total_limit 1 \
      || { log_fail "task=${task} model=${model_name} tokenizer=${tokenizer_name}"; continue; }

      # Optionally: touch a success marker
      touch "${outdir}/SUCCESS"
    done
  done
done
