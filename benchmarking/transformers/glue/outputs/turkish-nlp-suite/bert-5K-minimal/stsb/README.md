---
library_name: transformers
language:
- en
base_model: turkish-nlp-suite/bert-5K-minimal
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- spearmanr
model-index:
- name: stsb
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE STSB
      type: glue
      args: stsb
    metrics:
    - name: Spearmanr
      type: spearmanr
      value: 0.22467982113115487
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# stsb

This model is a fine-tuned version of [turkish-nlp-suite/bert-5K-minimal](https://huggingface.co/turkish-nlp-suite/bert-5K-minimal) on the GLUE STSB dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1656
- Pearson: 0.2233
- Spearmanr: 0.2247
- Combined Score: 0.2240

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.57.3
- Pytorch 2.5.1+cu121
- Datasets 4.4.1
- Tokenizers 0.22.1
