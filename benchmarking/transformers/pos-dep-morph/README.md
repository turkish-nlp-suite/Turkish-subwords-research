# Running benchmarking on BERT models on POS-DEP-MORPH task

This folder has 3 scripts, the main script is `train_pos_morph.py` which does the real training given the dataset and the transformer. For this task we train on BOUN Treebank. We fetched the train, dev test files from original [BOUN Treebank repo](https://github.com/UniversalDependencies/UD_Turkish-BOUN).

`run_pos_morph.sh` will run the BERTurk baseline on the treebank.

To run all the transformers of vocab and corpora sizes, navigate to `run_all` then run `run_all.sh`. `run_all.sh` downloads all the BERT models from their repos on [HF](https://huggingface.co/collections/turkish-nlp-suite/turkish-subwords-research). This script iterates over all the vocab sizes and corpora sizes. 

The batch size, seeds and number of epochs will reproduce the numbers on the research paper.

