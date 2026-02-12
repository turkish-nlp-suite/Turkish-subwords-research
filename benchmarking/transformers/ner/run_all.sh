#!/usr/bin/env bash
set -u  # keep undefined-vars safety

log_fail() {
  echo "[FAIL] $*" >&2
}

for vocab in 2 5 10 20 32 52 128; do
  for ds in minimal books alldata; do
    model_name="turkish-nlp-suite/bert-${vocab}K-${ds}"
    tokenizer_name="turkish-nlp-suite/wordpiece_${vocab}k_cased_${ds}"
    outdir="outputs/${model_name}"
    python3 train_ner.py  \
	    --model ${model_name} \
	    --tokenizer ${tokenizer_name} \
	    --dataset turkish-nlp-suite/turkish-wikiNER \
	    --text_column tokens \
	    --label_column tags \
	    --max_length 128 \
	    --epochs 10 \
            --batch_size 128 \
            --lr 3e-5 \
	    --seeds 42 43 44  \
	    --output_dir ${outdir} \

  done
done
