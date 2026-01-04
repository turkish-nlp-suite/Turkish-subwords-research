#!/usr/bin/env python
import argparse, os, random, json, numpy as np, torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset, ClassLabel
from transformers import (AutoConfig, AutoTokenizer, AutoModelForTokenClassification,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments,
                          set_seed)
from seqeval.metrics import f1_score as seqeval_f1, classification_report
from seqeval.scheme import IOB2

def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def align_labels_with_tokens(labels: List[int], word_ids: List[Optional[int]]) -> List[int]:
    # First-subword labeling: label on first piece, -100 on others
    previous_word_idx = None
    new_labels = []
    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append(-100)
        elif word_idx != previous_word_idx:
            new_labels.append(labels[word_idx])
        else:
            new_labels.append(-100)
        previous_word_idx = word_idx
    return new_labels

def decode_preds_to_spans(pred_ids: List[List[int]], label_ids: List[List[int]], id2label: Dict[int,str], word_ids_list: List[List[Optional[int]]]) -> Tuple[List[List[str]], List[List[str]]]:
    # Convert token-level BIO predictions back to word-level BIO for seqeval
    pred_tags_word, gold_tags_word = [], []
    for pred_row, gold_row, word_ids in zip(pred_ids, label_ids, word_ids_list):
        # Reconstruct per-word tags: take the first-subword tag per word (ignore -100)
        word_to_pred = {}
        word_to_gold = {}
        for tid, (p, g, w) in enumerate(zip(pred_row, gold_row, word_ids)):
            if w is None: continue
            if w not in word_to_pred: word_to_pred[w] = id2label.get(p, "O")
            if g != -100 and (w not in word_to_gold): word_to_gold[w] = id2label.get(g, "O")
        # Normalize to contiguous word order starting from 0
        max_w = max(word_to_pred.keys() | word_to_gold.keys()) if (word_to_pred or word_to_gold) else -1
        pred_seq, gold_seq = [], []
        for w in range(max_w+1):
            pred_seq.append(word_to_pred.get(w, "O"))
            gold_seq.append(word_to_gold.get(w, "O"))
        pred_tags_word.append(pred_seq)
        gold_tags_word.append(gold_seq)
    return pred_tags_word, gold_tags_word

def compute_metrics_builder(id2label: Dict[int,str], tokenizer):
    def compute_metrics(p):
        logits, labels = p
        preds = np.argmax(logits, axis=-1)
        # Need word_ids for each example to rebuild word-level tags
        # We stash them via the Trainer's dataset; here, we rely on a side channel set in preprocess
        word_ids_list = tokenizer._last_word_ids  # set during eval preprocess
        pred_tags_word, gold_tags_word = decode_preds_to_spans(preds.tolist(), labels.tolist(), id2label, word_ids_list)
        f1 = seqeval_f1(gold_tags_word, pred_tags_word, scheme=IOB2)
        return {"f1": f1}
    return compute_metrics

@dataclass
class ExampleWithWordIds:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    word_ids: List[Optional[int]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", default="conll2003", help="HF dataset name or local path")
    ap.add_argument("--subset", default=None, help="dataset config/subset if any (e.g., 'tr' for TRGLUE)")
    ap.add_argument("--text_column", default="tokens")
    ap.add_argument("--label_column", default="ner_tags")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--gradient_accumulation", type=int, default=1)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--output_dir", default="ner_out")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    raw = load_dataset(args.dataset, args.subset) if args.subset else load_dataset(args.dataset)
    label_list = open("tags.lst", "r").read().split("\n")
    label_list = [ll.strip() for ll in label_list if ll.strip()]
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}

    def to_ids(batch):
      batch["tags"] = [[label2id[t] for t in seq] for seq in batch["tags"]]
      return batch


    config = AutoConfig.from_pretrained(args.model, num_labels=len(label_list), id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model, config=config)

    def preprocess(examples):
        tokenized = tokenizer(examples[args.text_column], is_split_into_words=True, truncation=True, max_length=args.max_length)
        all_labels, all_word_ids = [], []
        for i, labels in enumerate(examples[args.label_column]):
            word_ids = tokenized.word_ids(batch_index=i)
            aligned = align_labels_with_tokens(labels, word_ids)
            all_labels.append(aligned)
            all_word_ids.append(word_ids)
        tokenized["labels"] = all_labels
        # For evaluation, keep word_ids accessible; we stash on tokenizer (hack but simple)
        tokenizer._last_word_ids = all_word_ids
        return tokenized

    raw = raw.map(to_ids, batched=True)
    with_features = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    results_by_seed = []
    for seed in args.seeds:
        set_seed(seed); set_global_seed(seed)
        outdir = os.path.join(args.output_dir, f"seed{seed}")
        os.makedirs(outdir, exist_ok=True)

        model = AutoModelForTokenClassification.from_pretrained(args.model, config=config)

        training_args = TrainingArguments(
            output_dir=outdir,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=max(4, args.batch_size),
            gradient_accumulation_steps=args.gradient_accumulation,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=args.fp16,
            bf16=args.bf16,
            report_to=["none"],
        )

        compute_metrics = compute_metrics_builder(id2label, tokenizer)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=with_features["train"],
            eval_dataset=with_features["validation"] if "validation" in with_features else with_features["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        if not args.eval_only:
            trainer.train()

        metrics = trainer.evaluate(with_features["test"])
        f1 = metrics["eval_f1"]
        results_by_seed.append(f1)

        # Save per-seed report
        preds_logits, labels, _ = trainer.predict(with_features["test"])
        preds = np.argmax(preds_logits, axis=-1).tolist()
        word_ids_list = tokenizer._last_word_ids
        pred_tags_word, gold_tags_word = decode_preds_to_spans(preds, labels.tolist(), id2label, word_ids_list)
        report = classification_report(gold_tags_word, pred_tags_word, scheme=IOB2)
        with open(os.path.join(outdir, "seqeval_report.txt"), "w") as f:
            f.write(report)
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump({"f1": f1}, f, indent=2)

    mean_f1 = float(np.mean(results_by_seed))
    std_f1 = float(np.std(results_by_seed))
    with open(args.output_dir + "final.txt", "w") as ofile:
        ofile.write(json.dumps({"mean_f1": mean_f1, "std_f1": std_f1, "seeds": args.seeds}, indent=2) + "\n")

        # LaTeX-friendly row snippet
        ofile.write(f"LaTeX row: {args.model} & {mean_f1:.2f} $\\pm$ {std_f1:.2f} \\\\")

if __name__ == "__main__":
    main()
