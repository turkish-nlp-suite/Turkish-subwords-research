---
library_name: transformers
language:
- en
base_model: turkish-nlp-suite/bert-128K-books
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- matthews_correlation
model-index:
- name: cola
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE COLA
      type: glue
      args: cola
    metrics:
    - name: Matthews Correlation
      type: matthews_correlation
      value: 0.07711028009024962
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# cola

This model is a fine-tuned version of [turkish-nlp-suite/bert-128K-books](https://huggingface.co/turkish-nlp-suite/bert-128K-books) on the GLUE COLA dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6466
- Matthews Correlation: 0.0771

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 128
- eval_batch_size: 128
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
