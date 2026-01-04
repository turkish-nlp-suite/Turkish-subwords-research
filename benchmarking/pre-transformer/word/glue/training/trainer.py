#!/usr/bin/env python3
# coding: utf-8
import os
import json
import argparse
from typing import Optional, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import evaluate

from data_loader import build_dataloaders


device="cuda:0"

# -----------------------------
# Model
# -----------------------------
class WordBiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_labels: int, hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.3, is_regression: bool = False):
        super().__init__()
        vocab_size, emb_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.from_numpy(embedding_matrix))
        # Fine-tune by default; pass --freeze_embeddings to freeze
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.is_regression = is_regression
        self.classifier = nn.Linear(hidden_size * 2, 1 if is_regression else num_labels)

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)      # (B, T, D)
        out, _ = self.lstm(x)              # (B, T, 2H)
        pooled, _ = torch.max(out, dim=1)  # (B, 2H)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)   # (B, C) or (B, 1)
        loss = None
        if labels is not None:
            if self.is_regression:
                loss = nn.MSELoss()(logits.view(-1), labels.view(-1))
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)
        return logits, loss


# -----------------------------
# Metrics
# -----------------------------
def get_metric_for_task(task_name: str):
    if task_name == "sst2":
        return evaluate.load("accuracy")
    elif task_name == "stsb":
        return evaluate.load("glue", "stsb")  # pearson, spearmanr
    else:
        return evaluate.load("glue", task_name)

def compute_metrics(task_name: str, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    metric = get_metric_for_task(task_name)
    if task_name == "stsb":
        return metric.compute(predictions=preds.reshape(-1), references=labels.reshape(-1))
    else:
        return metric.compute(predictions=preds, references=labels)


# -----------------------------
# Train/Eval loops
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        _, loss = model(input_ids=input_ids, labels=labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_loop(model, loader, device, task_name: str, is_regression: bool) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, loss = model(input_ids=input_ids, labels=labels)
        if is_regression:
            pred = logits.squeeze(-1).cpu().numpy()
        else:
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
        lab = labels.cpu().numpy()
        all_preds.append(pred)
        all_labels.append(lab)
        total_loss += loss.item() * input_ids.size(0)

    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    metrics = compute_metrics(task_name, preds, labels)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# -----------------------------
# Orchestration
# -----------------------------
def run_experiment(
    task_name: str,
    vocab_path: str,
    fasttext_path: str,
    max_seq_len: int,
    batch_size: int,
    num_epochs: int,
    lr: float,
    output_dir: str,
    hidden_size: int = 256,
    num_layers: int = 1,
    dropout: float = 0.3,
    seed: int = 42,
    cache_dir: Optional[str] = None,
    freeze_embeddings: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader, eval_loader, emb_matrix, vocab, num_labels, is_regression = build_dataloaders(
        task_name=task_name,
        vocab_path=vocab_path,
        fasttext_path=fasttext_path,
        max_seq_len=max_seq_len,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WordBiLSTMClassifier(
        embedding_matrix=emb_matrix,
        num_labels=num_labels,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        is_regression=is_regression,
    )
    if freeze_embeddings:
        model.embedding.weight.requires_grad = False
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs))

    # Default selection metric
    if task_name == "stsb":
        best_metric_name = "pearson"
    elif task_name == "mnli":
        best_metric_name = "accuracy_matched"
    else:
        best_metric_name = "accuracy"

    best_score = -1e9
    history = []

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        if isinstance(eval_loader, dict):
            # MNLI: evaluate both splits
            split_metrics: Dict[str, float] = {}
            for split_name, loader in eval_loader.items():
                m = eval_loop(model, loader, device, task_name, is_regression)
                # Prefix keys with split identifier
                for k, v in m.items():
                    split_metrics[f"{k}_{split_name}"] = v
            record = {"epoch": epoch, "train_loss": train_loss, **split_metrics}

            # Choose selection score based on matched
            score = split_metrics.get("accuracy_matched")
            if score is None:
                # Fallback to average of numeric matched metrics
                matched_vals = [v for k, v in split_metrics.items() if k.endswith("_matched") and isinstance(v, (int, float, np.floating))]
                score = float(np.mean(matched_vals)) if matched_vals else -1e9
        else:
            val_metrics = eval_loop(model, eval_loader, device, task_name, is_regression)
            record = {"epoch": epoch, "train_loss": train_loss, **val_metrics}
            score = val_metrics.get(best_metric_name)
            if score is None and task_name != "stsb":
                # Fallback to average of numeric metrics
                num_vals = [v for k, v in val_metrics.items() if isinstance(v, (int, float, np.floating))]
                score = float(np.mean(num_vals)) if num_vals else -1e9

        history.append(record)

        if score is not None and score > best_score:
            best_score = score
            torch.save(model.state_dict(), os.path.join(output_dir, "best.pt"))

        with open(os.path.join(output_dir, "metrics_history.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Final summary
    if isinstance(eval_loader, dict):
        summary = {
            "best_metric": "accuracy_matched",
            "best_score": best_score,
            "note": "Metrics include both matched and mismatched with suffixes.",
            "final_epoch": history[-1] if history else {},
        }
    else:
        summary = {
            "best_metric": best_metric_name,
            "best_score": best_score,
            "final_epoch": history[-1] if history else {},
        }

    with open(os.path.join(output_dir, "metrics_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Word-level BiLSTM on TrGLUE with Turkish fastText embeddings")
    p.add_argument("--task_name", type=str, required=True, help="Task: mnli, mrpc, qnli, qqp, rte, stsb, etc.")
    p.add_argument("--vocab_path", type=str, required=True, help="Path to vocab file for the chosen fraction")
    p.add_argument("--fasttext_path", type=str, required=True, help="Path to Turkish fastText .bin or .vec")
    p.add_argument("--max_seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--freeze_embeddings", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_experiment(
        task_name=args.task_name,
        vocab_path=args.vocab_path,
        fasttext_path=args.fasttext_path,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
        cache_dir=args.cache_dir,
        freeze_embeddings=args.freeze_embeddings,
    )
    print(json.dumps(summary, ensure_ascii=False))
