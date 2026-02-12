# Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data, Vocabulary, Morphology Interplay

This is the repo for the research paper “Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data, Vocabulary, Morphology Interplay.” This paper dissects how tokenizer vocabulary size and training corpus choices interact with Turkish morphology, offering matched‑budget comparisons across tokenizer families, a morphology‑aware diagnostic toolkit, and ready‑to‑run pipelines for fully reproducible results.


## Code
The code somes in 2 parts, calculation of morphological statistics for the tokenizers and benchmarking the resulting models and vocab types. Please jump to the related subsection below to run the code, related subfolders include their own READMEs for directions to run the code directly.


### Morpho stats

### Benchmarking
Vocab types, hence the models are covered in 2 parts as well, pre-Tranformer era and Transformer era - WordPiece tokenizers. In the pre-Transformer era you'll see benchmarking code for character-level, word-level and subword-level vocabularies and models. 

* Pre-Tranformer era includes:

- Character-level vocab: This type of vocab is fixed and includes uppercase and lowercase Turkish letters along with digits and some punctuation marks. A typical tokenization is `g ##i ##t ##t ##i ##m`.
- Morpho-subword vocab: For this type of vocab, the words are segmented into subwords that are morphemes with morphological tools e.g. `gittim` -> `git ##ti ##m`. For more details, please refer to the paper. 
- Word-level vocab: For this type of vocab, each word joins the vocab as itself. 

All the vocabs above are **CASED**. For the char level vocab, models on GLUE are CNN models stacked on top of a character-based embedding layer. For NER the model comes with a char embedding layer, then a BiLSTM and CRF. Treebank parser POS-DEP-Morph again built on top of an char level embedding layer with biaffine layers, for the architecture please refer to the paper. For word-level and subword-level models, GLUE models are LSTM based models as well, all the other architectures are same with word/subword level mebeddings as well.

* Transformer modelling is built as BERT models with WordPiece tokenizers. For each vocab size 2K, 5K, 10K, 20K, 32K, 52K and 128K we trained tokenizers on 3 different sized corpora, Minimal, Medium and Alldata. One can run all the benchmarking with a single script per task, providing vocab size and corpus size as arguments. Please navigate to the `transformers` subfolder for more info.

## Hugging Face
All the tokenizers and BERT models can be found on the project [HF repo](https://huggingface.co/collections/turkish-nlp-suite/turkish-subwords-research).

## Blog post

## Research paper and citation
Preprint is available at [Arxiv]i(https://arxiv.org/abs/2602.06942). 

Cite the preprint:

```
@misc{altinok2026optimalturkishsubwordstrategies,
      title={Optimal Turkish Subword Strategies at Scale: Systematic Evaluation of Data, Vocabulary, Morphology Interplay}, 
      author={Duygu Altinok},
      year={2026},
      eprint={2602.06942},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.06942}, 
}
```

