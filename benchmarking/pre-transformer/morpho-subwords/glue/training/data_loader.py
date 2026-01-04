#!/usr/bin/env python3
# coding: utf-8
import os
from typing import List, Tuple, Dict, Optional, Union
import re, string
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_vectors

def kill_punct(text):
  punct = string.punctuation
  text = text.translate(text.maketrans(' ', ' ', punct))
  text = " ".join(text.strip().split())
  return text

# -----------------------------
# Tokenization (word-level)
# -----------------------------
def simple_word_tokenize(text: str) -> List[str]:
    # Lowercase + basic punctuation splitting, keeps Turkish characters
    text = text.strip().lower()
    text = kill_punct(text)
    # Replace non-letter/digit/apostrophe characters with space
    # (keeps Turkish diacritics thanks to \w under re.UNICODE)
    tokens = text.split()
    return tokens





# -----------------------------
# Dataset
# -----------------------------
class WordLevelGlueDataset(Dataset):
    def __init__(
        self,
        hf_split,
        sentence1_key: str,
        sentence2_key: Optional[str],
        vocab: Dict[str, int],
        max_seq_len: int,
        is_regression: bool,
    ):
        self.data = hf_split
        self.s1 = sentence1_key
        self.s2 = sentence2_key
        self.vocab = vocab
        self.unk_id = vocab.get("<unk>", 1)
        self.pad_id = vocab.get("<pad>", 0)
        self.sep_token = "<sep>"
        self.sep_id = vocab.get(self.sep_token, 2)
        self.max_seq_len = max_seq_len
        self.is_regression = is_regression

    def __len__(self):
        return len(self.data)

    def encode_text(self, text: str) -> List[int]:
        toks = simple_word_tokenize(text)
        ids = [self.vocab.get(t, self.unk_id) for t in toks]
        if len(ids) > self.max_seq_len:
            ids = ids[: self.max_seq_len]
        else:
            ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        return ids

    def __getitem__(self, idx):
        ex = self.data[idx]
        if self.s2 is None:
            text = ex[self.s1]
        else:
            t1 = ex[self.s1] if ex[self.s1] is not None else ""
            t2 = ex[self.s2] if ex[self.s2] is not None else ""
            # Soft concatenate with a visible separator token
            text = f"{t1} {self.sep_token} {t2}"

        input_ids = torch.tensor(self.encode_text(text), dtype=torch.long)

        if "label" in ex and ex["label"] is not None:
            if self.is_regression:
                label = torch.tensor(float(ex["label"]), dtype=torch.float)
            else:
                label = torch.tensor(int(ex["label"]), dtype=torch.long)
        else:
            label = None

        return {"input_ids": input_ids, "labels": label}


# -----------------------------
# Vocab and embeddings
# -----------------------------
def load_vocab_from_file(vocab_path: str, add_specials: bool = True) -> Dict[str, int]:
    """
    One token per line. We add specials:
      <pad>=0, <unk>=1, <sep>=2
    """
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    tokens = []
    with open(vocab_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.strip()
            if tok:
                tokens.append(tok)

    vocab: Dict[str, int] = {}
    idx = 0
    if add_specials:
        vocab["<pad>"] = idx; idx += 1
        vocab["<unk>"] = idx; idx += 1
        vocab["<sep>"] = idx; idx += 1

    for t in tokens:
        if t not in vocab:
            vocab[t] = idx
            idx += 1
    return vocab


def build_embedding_matrix(
    vocab: Dict[str, int],
    fasttext_path: str,
    dim: int = 300,
) -> np.ndarray:
    """
    Loads Turkish fastText vectors (.bin via Facebook format or .vec in word2vec txt)
    and aligns to vocab indices. Unknowns are random N(0, 0.02). <pad> is zero.
    """
    if not os.path.exists(fasttext_path):
        raise FileNotFoundError(f"fastText vectors not found: {fasttext_path}")

    if fasttext_path.endswith(".bin"):
        kv = load_facebook_vectors(fasttext_path)
    else:
        kv = KeyedVectors.load_word2vec_format(fasttext_path, binary=False)

    matrix = np.random.normal(scale=0.02, size=(len(vocab), dim)).astype(np.float32)

    pad_id = vocab.get("<pad>", None)
    if pad_id is not None:
        matrix[pad_id] = np.zeros(dim, dtype=np.float32)

    for token, idx in vocab.items():
        if token in ("<pad>", "<unk>", "<sep>"):
            continue
        if token in kv.key_to_index:
            matrix[idx] = kv[token]
        else:
            lower = token.lower()
            if lower in kv.key_to_index:
                matrix[idx] = kv[lower]
            # else: keep random init

    return matrix


# -----------------------------
# Task mapping and loaders
# -----------------------------
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def load_trglue_splits(task_name: str, cache_dir: Optional[str] = None):
    return load_dataset("turkish-nlp-suite/TrGLUE", task_name, cache_dir=cache_dir)


def build_dataloaders(
    task_name: str,
    vocab_path: str,
    fasttext_path: str,
    max_seq_len: int,
    batch_size: int,
    cache_dir: Optional[str] = None,
    num_workers: int = 2,
) -> Tuple[DataLoader, Union[DataLoader, Dict[str, DataLoader]], np.ndarray, Dict[str, int], int, bool]:
    """
    Returns:
      - train_loader
      - eval_loader (DataLoader or dict(split->DataLoader) for MNLI)
      - embedding_matrix (np.ndarray)
      - vocab (dict)
      - num_labels (int)
      - is_regression (bool)
    """
    raw = load_trglue_splits(task_name, cache_dir=cache_dir)
    s1, s2 = TASK_TO_KEYS[task_name]

    is_regression = task_name == "stsb"
    if is_regression:
        num_labels = 1
    else:
        label_names = raw["train"].features["label"].names
        num_labels = len(label_names)

    vocab = load_vocab_from_file(vocab_path, add_specials=True)
    emb_matrix = build_embedding_matrix(vocab, fasttext_path, dim=300)

    train_ds = WordLevelGlueDataset(raw["train"], s1, s2, vocab, max_seq_len, is_regression)

    # Validation handling: MNLI has matched + mismatched; others have validation.
    if task_name == "mnli":
        eval_ds_matched = WordLevelGlueDataset(raw["validation_matched"], s1, s2, vocab, max_seq_len, is_regression)
        eval_ds_mismatched = WordLevelGlueDataset(raw["validation_mismatched"], s1, s2, vocab, max_seq_len, is_regression)
    else:
        eval_ds = WordLevelGlueDataset(raw["validation"], s1, s2, vocab, max_seq_len, is_regression)

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels_list = [b["labels"] for b in batch]
        if labels_list[0] is None:
            labels = None
        else:
            if is_regression:
                labels = torch.stack(labels_list, dim=0).float()
            else:
                labels = torch.stack(labels_list, dim=0).long()
        return {"input_ids": input_ids, "labels": labels}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn
    )

    if task_name == "mnli":
        eval_loader: Union[DataLoader, Dict[str, DataLoader]] = {
            "matched": DataLoader(
                eval_ds_matched, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
            ),
            "mismatched": DataLoader(
                eval_ds_mismatched, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
            ),
        }
    else:
        eval_loader = DataLoader(
            eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
        )

    return train_loader, eval_loader, emb_matrix, vocab, num_labels, is_regression
