# Running benchmarking on BERT models on NER dataset

This folder has 3 scripts, the main script is `train_ner.py` which does the NER training given the NER dataset and the transformer. For this task we train on Turkis-WikiNER dataset.

`run_ner.sh` will run the BERTurk baseline on the chosen NER set.

`run_all.sh` downloads all the BERT models from their repos on [HF](https://huggingface.co/collections/turkish-nlp-suite/turkish-subwords-research). This script iterates over all the vocab sizes and corpora sizes. 

The batch size, seeds and number of epochs will reproduce the numbers on the research paper.

