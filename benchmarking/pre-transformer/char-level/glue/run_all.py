import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset
import evaluate

# -----------------------
# Utils
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------
# Char vocab + encoding
# -----------------------
def build_char_vocab(extra_chars: str = "", lower: bool = True) -> Dict[str, int]:
    base = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,;:!?\'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    )
    alphabet = sorted(set(base + extra_chars))
    if lower:
        alphabet = sorted(set("".join(ch.lower() for ch in alphabet)))
    tokens = ["<pad>", "<unk>", "<sep>"]
    stoi = {ch: i + len(tokens) for i, ch in enumerate(alphabet)}
    for i, t in enumerate(tokens):
        stoi[t] = i
    return stoi

def encode_text_to_chars(text: str, stoi: Dict[str, int], max_len: int, lower: bool = True) -> List[int]:
    if text is None:
        text = ""
    if lower:
        text = text.lower()
    unk = stoi["<unk>"]
    ids = []
    for ch in text:
        ids.append(stoi.get(ch, unk))
        if len(ids) >= max_len:
            break
    return ids

# -----------------------
# HF collator
# -----------------------
@dataclass
class CharCollator:
    stoi: Dict[str, int]
    max_len: int = 400
    lower: bool = True
    is_pair: bool = False
    task: str = "sst2"  # used for routing in the loop

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        pad_id = self.stoi["<pad>"]
        sep_id = self.stoi["<sep>"]
        seqs = []
        lengths = []
        # Labels or scores
        if self.task == "stsb":
            scores = torch.tensor([float(ex["label"]) for ex in batch], dtype=torch.float32)
            y_out = {"scores": scores}
        else:
            labels = torch.tensor([int(ex["label"]) for ex in batch], dtype=torch.long)
            y_out = {"labels": labels}

        for ex in batch:
            if self.is_pair:
                a = ex["text_a"]
                b = ex["text_b"]
                half = (self.max_len - 1) // 2
                a_ids = encode_text_to_chars(a, self.stoi, half, self.lower)
                rem = self.max_len - 1 - len(a_ids)
                b_ids = encode_text_to_chars(b, self.stoi, rem, self.lower)
                ids = a_ids + [sep_id] + b_ids
            else:
                ids = encode_text_to_chars(ex["text"], self.stoi, self.max_len, self.lower)
            seqs.append(ids)
            lengths.append(len(ids))

        B = len(seqs)
        T = max(lengths) if lengths else 1
        padded = torch.full((B, T), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            padded[i, : len(s)] = torch.tensor(s, dtype=torch.long)

        out = {
            "input_ids": padded,
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }
        out.update(y_out)
        return out

# -----------------------
# Model
# -----------------------
class CharCNNBackbone(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 64, conv_channels: int = 256,
                 kernel_sizes: Tuple[int, ...] = (3, 5, 7), dropout: float = 0.2,
                 pad_idx: int = 0, pooling: str = "max"):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, conv_channels, k, padding=k // 2) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.pooling = pooling
        self.out_dim = conv_channels * len(kernel_sizes)
        nn.init.uniform_(self.embedding.weight, -0.05, 0.05)
        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()
        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))
            nn.init.zeros_(conv.bias)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)  # [B, T, E]
        x = x.transpose(1, 2)          # [B, E, T]
        feats = []
        for conv in self.convs:
            h = torch.relu(conv(x))    # [B, C, T]
            if self.pooling == "max":
                h = F.max_pool1d(h, h.size(-1)).squeeze(-1)  # [B, C]
            else:
                h = h.mean(dim=-1)     # [B, C]
            feats.append(h)
        h = torch.cat(feats, dim=-1)   # [B, C*k]
        h = self.dropout(h)
        return h

class MultiTaskCharModel(nn.Module):
    def __init__(self, vocab_size: int, num_labels: Dict[str, int], pad_idx: int = 0):
        super().__init__()
        self.backbone = CharCNNBackbone(vocab_size=vocab_size, pad_idx=pad_idx)
        self.classifiers = nn.ModuleDict()
        self.regressors = nn.ModuleDict()
        for task, n in num_labels.items():
            if task == "stsb":
                self.regressors[task] = nn.Linear(self.backbone.out_dim, 1)
                nn.init.zeros_(self.regressors[task].bias)
            else:
                self.classifiers[task] = nn.Linear(self.backbone.out_dim, n)
                nn.init.zeros_(self.classifiers[task].bias)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor, task: str):
        h = self.backbone(input_ids, lengths)
        if task == "stsb":
            return self.regressors[task](h).squeeze(-1)
        else:
            return self.classifiers[task](h)

# -----------------------
# Task configs (HF schemas)
# -----------------------
TASKS = {
    "cola": {
        "hf_name": "turkish-nlp-suite/TrGLUE",
        "fields": ("sentence", None),
        "num_labels": 2,
        "is_pair": False,
        "metric": "matthews_correlation",
    },
    "sst2": {
        "hf_name": "turkish-nlp-suite/TrGLUE",
        "subset": "sst2",
        "fields": ("sentence", None),
        "num_labels": 2,
        "is_pair": False,
        "metric": "accuracy",
    },
    "mnli": {
        "hf_name": "turkish-nlp-suite/TrGLUE",
        "subset": "mnli",
        "fields": ("premise", "hypothesis"),
        "num_labels": 3,
        "is_pair": True,
        "metric": "accuracy",
    },
    "mrpc": {
        "hf_name": "turkish-nlp-suite/TrGLUE",
        "subset": "mrpc",
        "fields": ("sentence1", "sentence2"),
        "num_labels": 2,
        "is_pair": True,
        "metric": "accuracy",  # you can also compute F1
    },
    "stsb": {
        "hf_name": "turkish-nlp-suite/TrGLUE",
        "subset": "stsb",
        "fields": ("sentence1", "sentence2"),
        "num_labels": 1,
        "is_pair": True,
        "metric": "pearson",  # we’ll also report spearman
    },
}

# -----------------------
# Dataset preprocessing to unified dicts
# -----------------------
def to_unified_format(example, fields, task_name):
    a, b = fields
    if b is None:
        return {"text": example[a], "label": int(example["label"])}
    else:
        if task_name == "stsb":
            return {"text_a": example[a], "text_b": example[b], "label": float(example["label"])}
        else:
            return {"text_a": example[a], "text_b": example[b], "label": int(example["label"])}

def load_hf_splits(task_cfg, task_name):
    ds = load_dataset(task_cfg["hf_name"], task_name)
    fields = task_cfg["fields"]
    map_fn = lambda ex: to_unified_format(ex, fields, task_name)
    ds = ds.map(map_fn, remove_columns=ds["train"].column_names)
    # mnli has validation_matched and validation_mismatched
    if task_name == "mnli":
        train = ds["train"]
        val_mat = ds["validation_matched"]
        val_mis = ds["validation_mismatched"]
        test_mat = ds.get("test_matched")
        test_mis = ds.get("test_mismatched")
        return train, {"matched": val_mat, "mismatched": val_mis}, {"matched": test_mat, "mismatched": test_mis}
    elif task_name == "stsb":
        return ds["train"], {"default": ds["validation"]}, {"default": ds.get("validation")}
    else:
        return ds["train"], {"default": ds["validation"]}, {"default": ds.get("test")}

# -----------------------
# Metrics wrappers
# -----------------------
def build_metric(task_name: str):
    if task_name == "cola":
        return evaluate.load("matthews_correlation")
    elif task_name in ["sst2", "mnli", "mrpc"]:
        return evaluate.load("accuracy")
    elif task_name == "stsb":
        return {
            "pearson": evaluate.load("pearsonr"),
            "spearman": evaluate.load("spearmanr"),
        }
    else:
        return None

def compute_eval(task_name: str, logits_or_scores, labels_or_scores):
    if task_name == "stsb":
        preds = logits_or_scores
        ref = labels_or_scores
        pear = evaluate.load("pearsonr")
        spear = evaluate.load("spearmanr")
        pear_r = pear.compute(references=ref, predictions=preds)["pearsonr"]
        spear_r = spear.compute(references=ref, predictions=preds)["spearmanr"]
        return {"pearson": pear_r, "spearman": spear_r}
    else:
        preds = logits_or_scores.argmax(-1)
        acc = evaluate.load("accuracy").compute(references=labels_or_scores, predictions=preds)["accuracy"]
        if task_name == "mrpc":
            f1 = evaluate.load("f1").compute(references=labels_or_scores, predictions=preds)["f1"]
            return {"accuracy": acc, "f1": f1}
        if task_name == "cola":
            mcc = evaluate.load("matthews_correlation").compute(references=labels_or_scores, predictions=preds)["matthews_correlation"]
            return {"mcc": mcc}
        return {"accuracy": acc}

# -----------------------
# Training & eval loops
# -----------------------
def train_one_epoch(model, dataloader, task_name, device, optimizer, scheduler=None, is_regression=False):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if is_regression:
            scores = batch["scores"].to(device)
            preds = model(input_ids, lengths, task_name)
            loss = F.mse_loss(preds, scores)
        else:
            labels = batch["labels"].to(device)
            logits = model(input_ids, lengths, task_name)
            loss = F.cross_entropy(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / max(1, len(dataloader))

@torch.no_grad()
def evaluate_split(model, dataloader, task_name, device, is_regression=False):
    model.eval()
    preds_all = []
    refs_all = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        if is_regression:
            scores = batch["scores"].cpu().numpy()
            out = model(input_ids, lengths, task_name).cpu().numpy()
            preds_all.append(out)
            refs_all.append(scores)
        else:
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids, lengths, task_name).cpu().numpy()
            preds_all.append(logits)
            refs_all.append(labels)
    preds = np.concatenate(preds_all, axis=0)
    refs = np.concatenate(refs_all, axis=0)
    return compute_eval(task_name, preds, refs)

# -----------------------
# Build loaders per task
# -----------------------
def make_loaders(task_name: str, stoi: Dict[str, int], batch_size: int = 128, max_len: int = 400, num_workers: int = 2):
    cfg = TASKS[task_name]
    train_ds, val_splits, _ = load_hf_splits(cfg, task_name)

    def ds_to_list(ds, cfg, task_name):
        items = []
        if cfg["is_pair"]:
            for ex in ds:
                items.append({"text_a": ex["text_a"], "text_b": ex["text_b"], "label": ex["label"]})
        else:
            for ex in ds:
                items.append({"text": ex["text"], "label": ex["label"]})
        return items

    # Convert to simple python lists for our collator
    train_items = ds_to_list(train_ds, cfg, task_name)
    val_items = {k: ds_to_list(v, cfg, task_name) for k, v in val_splits.items()}

    from torch.utils.data import Dataset
    class ListDS(Dataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]

    collator = CharCollator(
        stoi=stoi,
        max_len=max_len,
        lower=True,
        is_pair=cfg["is_pair"],
        task="stsb" if task_name == "stsb" else task_name
    )

    train_loader = DataLoader(
        ListDS(train_items),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loaders = {
        split: DataLoader(ListDS(items), batch_size=batch_size, shuffle=False, collate_fn=collator,
                          num_workers=num_workers, pin_memory=True)
        for split, items in val_items.items()
    }
    return train_loader, val_loaders

# -----------------------
# Main
# -----------------------
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build vocab and model
    stoi = build_char_vocab(lower=True)
    pad_idx = stoi["<pad>"]
    num_labels = {
        "cola": 2,
        "sst2": 2,
        "mnli": 3,
        "mrpc": 2,
        "stsb": 1,
    }
    model = MultiTaskCharModel(vocab_size=len(stoi), num_labels=num_labels, pad_idx=pad_idx).to(device)

    # Hyperparams
    batch_size = 128
    max_len = 400
    lr = 2e-3
    weight_decay = 0.01
    epochs = 5

    # Train & eval per task independently (shared architecture; you can also multi-task train)
    results = {}
    for task in ["cola", "sst2", "mnli", "mrpc", "stsb"]:
        print(f"\n=== Task: {task} ===")
        train_loader, val_loaders = make_loaders(task, stoi, batch_size=batch_size, max_len=max_len)

        # Fresh optimizer per task (or reuse if you want continual finetuning)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_metric = -1e9
        best_epoch = -1
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model, train_loader, task, device, optimizer,
                scheduler=None, is_regression=(task == "stsb")
            )
            print(f"Epoch {epoch} | train loss: {train_loss:.4f}")

            # Evaluate on all validation splits
            val_scores = {}
            for split, vloader in val_loaders.items():
                metrics = evaluate_split(
                    model, vloader, task, device, is_regression=(task == "stsb")
                )
                val_scores[split] = metrics
            print("Val:", val_scores)

            # Track best by primary metric
            if task == "cola":
                curr = np.mean([m["mcc"] for m in val_scores.values()])
            elif task == "stsb":
                curr = np.mean([m["pearson"] for m in val_scores.values()])
            else:
                # accuracy for sst2/mnli/mrpc
                accs = []
                for m in val_scores.values():
                    if "accuracy" in m:
                        accs.append(m["accuracy"])
                curr = float(np.mean(accs)) if accs else -1e9

            if curr > best_metric:
                best_metric = curr
                best_epoch = epoch

        results[task] = {"best": best_metric, "epoch": best_epoch}

    print("\nSummary (best val metrics):")
    for t, v in results.items():
        print(f"{t}: {v}")

if __name__ == "__main__":
    main()
