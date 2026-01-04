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
    outdir="outputs/${model_name}"
    python3 train_pos_morph.py  \
	    --model ${model_name} \
	    --tokenizer ${tokenizer_name} \
	    --train tr_boun-ud-train.conllu \
	    --dev tr_boun-ud-dev.conllu \
	    --test tr_boun-ud-test.conllu \
	    --max_length 64 \
	    --epochs 10 \
            --batch_size 64 \
            --lr 3e-5 \
	    --seeds 42  \
	    --output_dir ${outdir} \

  done
done
