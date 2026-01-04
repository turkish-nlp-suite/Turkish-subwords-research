python3 train_pos_morph.py \
    --model dbmdz/bert-base-turkish-cased \
	--train tr_boun-ud-train.conllu \
	--dev tr_boun-ud-dev.conllu \
	--test tr_boun-ud-test.conllu \
	--max_length 64 \
	--epochs 10 \
	--batch_size 64 \
	--lr 3e-5 \
	--seeds 42 
