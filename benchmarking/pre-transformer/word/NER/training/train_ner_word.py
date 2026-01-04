#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json, random, sys
from typing import List, Tuple, Dict, Optional
from collections import Counter

import numpy as np
import torch, string
from datasets import load_dataset
from seqeval.metrics import f1_score as seqeval_f1, classification_report
from seqeval.scheme import IOB2

# ===================
# Repro and device
# ===================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================
# Vocab utilities
# ===================
PAD = "<pad>"
UNK = "<unk>"

def kill_punct(text):
    punct = string.punctuation
    text = text.translate(text.maketrans(' ', ' ', punct))
    text = " ".join(text.strip().split())
    return text

def normalize_token(token, lower=False):
    token = kill_punct(token)
    if lower:
        token = token.lower()
    return token

def load_vocab_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        itos = [line.rstrip("\n") for line in f]
    itos = [PAD, UNK] + itos
    assert len(itos) >= 2 and itos[0] == PAD and itos[1] == UNK, "vocab file must start with <pad> and <unk>"
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def save_vocab_file(itos: List[str], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for w in itos:
            f.write(w + "\n")

def build_word_vocab(datasets, text_col="tokens", max_vocab=50000, min_freq=1, lower=False, normalize_digits=False):
    cnt = Counter()
    for ex in datasets["train"]:
        toks = [normalize_token(t, lower) for t in ex[text_col]]
        cnt.update(toks)
    itos = [PAD, UNK]
    items = [(w, c) for w, c in cnt.items() if c >= min_freq and w not in itos]
    items.sort(key=lambda x: (-x[1], x[0]))
    for w, _ in items[: max(0, max_vocab - len(itos))]:
        itos.append(w)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

# ===================
# Label utilities
# ===================
def load_tags_list(path="tags.lst"):
    with open(path, "r", encoding="utf-8") as f:
        tag_list = [l.strip() for l in f if l.strip()]
    assert "O" in tag_list, "tags.lst must contain 'O'"
    id2label = {i: l for i, l in enumerate(tag_list)}
    label2id = {l: i for i, l in enumerate(tag_list)}
    return tag_list, id2label, label2id

def tags_to_ids(tags: List[str], label2id: Dict[str, int]) -> List[int]:
    return [label2id.get(t, label2id["O"]) for t in tags]

# ===================
# Padding and batching
# ===================
def pad_batch(seqs, pad=0):
    maxlen = max((len(s) for s in seqs), default=0)
    out = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
    for i, s in enumerate(seqs):
        L = len(s)
        if L > 0:
            out[i, :L] = torch.tensor(s, dtype=torch.long)
            mask[i, :L] = True
    return out, mask

def batchify(x_pad, y_pad, mask, batch_size):
    N = x_pad.size(0)
    for i in range(0, N, batch_size):
        yield x_pad[i:i+batch_size], y_pad[i:i+batch_size], mask[i:i+batch_size]

# ===================
# CRF
# ===================
class CRF(torch.nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))
        torch.nn.init.uniform_(self.transitions, -0.1, 0.1)

    def neg_log_likelihood(self, emissions, tags, mask):
        log_den = self._compute_log_partition(emissions, mask)
        log_num = self._compute_joint_log_likelihood(emissions, tags, mask)
        return torch.mean(log_den - log_num)

    def _compute_joint_log_likelihood(self, emissions, tags, mask):
        B, T, C = emissions.size()
        score = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        trans_score = torch.zeros(B, device=emissions.device)
        for t in range(1, T):
            prev = tags[:, t-1]
            curr = tags[:, t]
            m = mask[:, t] & mask[:, t-1]
            trans_score[m] += self.transitions[prev[m], curr[m]]
        score = (score * mask).sum(dim=1) + trans_score
        return score

    def _compute_log_partition(self, emissions, mask):
        B, T, C = emissions.size()
        log_alpha = emissions[:, 0]
        for t in range(1, T):
            emit_t = emissions[:, t].unsqueeze(1)           # [B,1,C]
            trans = self.transitions.unsqueeze(0)           # [1,C,C]
            scores = log_alpha.unsqueeze(2) + trans + emit_t
            log_alpha_next = torch.logsumexp(scores, dim=1) # [B,C]
            log_alpha = torch.where(mask[:, t].unsqueeze(1), log_alpha_next, log_alpha)
        return torch.logsumexp(log_alpha, dim=1)

    def decode(self, emissions, mask):
        B, T, C = emissions.size()
        backpointers = []
        v = emissions[:, 0]
        for t in range(1, T):
            emit_t = emissions[:, t].unsqueeze(1)
            scores = v.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t
            v_t, bp = torch.max(scores, dim=1)
            upd = mask[:, t].unsqueeze(1)
            v = torch.where(upd, v_t, v)
            backpointers.append(bp)
        best_last = torch.argmax(v, dim=1)
        paths = []
        for b in range(B):
            seq = [best_last[b].item()]
            for t in reversed(backpointers):
                seq.append(t[b, seq[-1]].item())
            seq.reverse()
            L = int(mask[b].sum().item())
            paths.append(seq[:L])
        return paths

# ===================
# Model
# ===================
class WordBiLSTMCRF(torch.nn.Module):
    def __init__(self, num_words, num_tags, emb_dim=300, hidden=256, num_layers=1, dropout=0.3,
                 proj_dim=None, padding_idx=0, pretrained_weight: Optional[torch.Tensor]=None,
                 freeze_emb: bool=False):
        super().__init__()
        self.word_emb = torch.nn.Embedding(num_words, emb_dim, padding_idx=padding_idx)
        if pretrained_weight is not None:
            if pretrained_weight.shape != self.word_emb.weight.shape:
                raise ValueError(f"pretrained_weight shape {tuple(pretrained_weight.shape)} != expected {tuple(self.word_emb.weight.shape)}")
            with torch.no_grad():
                self.word_emb.weight.copy_(pretrained_weight)
        else:
            torch.nn.init.xavier_uniform_(self.word_emb.weight)
            with torch.no_grad():
                self.word_emb.weight[padding_idx].zero_()
        self.word_emb.weight.requires_grad_(not freeze_emb)

        self.proj = None
        input_dim = emb_dim
        if proj_dim is not None and proj_dim != emb_dim:
            self.proj = torch.nn.Linear(emb_dim, proj_dim)
            input_dim = proj_dim

        self.lstm = torch.nn.LSTM(
            input_size=input_dim, hidden_size=hidden // 2, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, x, mask, tags=None):
        emb = self.word_emb(x)
        if self.proj is not None:
            emb = self.proj(emb)
        emb = self.dropout(emb)
        out, _ = self.lstm(emb)
        logits = self.fc(self.dropout(out))
        if tags is not None:
            loss = self.crf.neg_log_likelihood(logits, tags, mask)
            return {"loss": loss, "logits": logits}
        else:
            pred = self.crf.decode(logits, mask)
            return {"pred_ids": pred, "logits": logits}

# ===================
# Data prep
# ===================
def build_dataset(split_ds, word2id, id2label, label2id, text_col="tokens", label_col="tags",
                  lower=False, normalize_digits=False):
    X, Y = [], []
    for ex in split_ds:
        tokens = [normalize_token(t, lower) for t in ex[text_col]]
        tag_raw = ex[label_col]
        tags = [id2label[i] if isinstance(i, int) else str(i) for i in tag_raw]
        word_ids = [word2id.get(w, word2id[UNK]) for w in tokens]
        tag_ids = tags_to_ids(tags, label2id)
        X.append(word_ids); Y.append(tag_ids)
    x_pad, mask = pad_batch(X, pad=word2id[PAD])
    y_pad, _ = pad_batch(Y, pad=label2id["O"])
    return x_pad, y_pad, mask

# ===================
# FT loading
# ===================
def load_fasttext_vectors(path: str):
    if path.endswith(".vec"):
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load_word2vec_format(path, binary=False)
        return ("gensim", kv, kv.vector_size)
    elif path.endswith(".bin"):
        import fasttext  # pip install fasttext
        ft = fasttext.load_model(path)
        return ("fasttext", ft, ft.get_dimension())
    else:
        raise ValueError("FastText file must be .vec or .bin")

def get_ft_vector(ft_obj, backend, token: str):
    if backend == "gensim":
        return ft_obj[token] if token in ft_obj else None
    else:
        try:
            return ft_obj.get_word_vector(token)
        except Exception:
            return None

def build_pretrained_weight_from_fasttext(id2word: List[str], ft_path: str, pad_token=PAD, unk_token=UNK,
                                          oov_policy="random"):
    backend, ft, ft_dim = load_fasttext_vectors(ft_path)
    V = len(id2word)
    W = torch.empty(V, ft_dim)
    torch.nn.init.xavier_uniform_(W)

    with torch.no_grad():
        # PAD
        if pad_token in id2word:
            W[id2word.index(pad_token)] = 0.0
        # UNK
        if unk_token in id2word:
            torch.nn.init.normal_(W[id2word.index(unk_token)], mean=0.0, std=0.02)
        # Fill others
        for i, tok in enumerate(id2word):
            if tok in (pad_token, unk_token):
                continue
            vec = get_ft_vector(ft, backend, tok)
            if vec is None:
                if oov_policy == "zero":
                    W[i] = 0.0
                else:
                    torch.nn.init.normal_(W[i], mean=0.0, std=0.02)
            else:
                W[i] = torch.tensor(vec, dtype=W.dtype)
    return W, ft_dim

# Back-compat: in-place initialization method (optional)
def init_embeddings_from_fasttext(model: WordBiLSTMCRF, word2id: Dict[str,int], id2word: List[str],
                                  ft_path: str, oov_policy="random", freeze=False, verbose=True):
    backend, ft, ft_dim = load_fasttext_vectors(ft_path)
    emb = model.word_emb
    if emb.embedding_dim != ft_dim and model.proj is None:
        raise ValueError(f"Embedding dim ({emb.embedding_dim}) != FastText dim ({ft_dim}). "
                         f"Either set --emb_dim={ft_dim} or use --proj_dim to project.")
    with torch.no_grad():
        weight = emb.weight
        weight[word2id[PAD]].zero_()
        torch.nn.init.normal_(weight[word2id[UNK]], mean=0.0, std=0.02)
        for i, tok in enumerate(id2word):
            if i < 2:  # PAD, UNK handled
                continue
            vec = get_ft_vector(ft, backend, tok)
            if vec is None:
                if oov_policy == "zero":
                    weight[i].zero_()
                else:
                    torch.nn.init.normal_(weight[i], mean=0.0, std=0.02)
            else:
                weight[i].copy_(torch.tensor(vec, dtype=weight.dtype))
    if freeze:
        emb.weight.requires_grad_(False)
    if verbose:
        print(f"[FastText] init done: vocab={len(id2word)}, dim={ft_dim}, frozen={freeze}, oov_policy={oov_policy}")

# ===================
# Train/Eval loops
# ===================
def train_one_epoch(model, loader, opt, device, clip=5.0):
    model.train()
    total = 0.0
    for x, y, m in loader:
        x = x.to(device); y = y.to(device); m = m.to(device)
        out = model(x, m, tags=y)
        loss = out["loss"]
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, id2label, device):
    model.eval()
    all_pred, all_gold = [], []
    for x, y, m in loader:
        x = x.to(device); y = y.to(device); m = m.to(device)
        out = model(x, m, tags=None)
        preds = out["pred_ids"]  # list of lists
        gold = [seq[:int(ms.sum().item())].tolist() for seq, ms in zip(y, m)]
        for p_ids, g_ids in zip(preds, gold):
            all_pred.append([id2label[i] for i in p_ids])
            all_gold.append([id2label[i] for i in g_ids])
    f1 = seqeval_f1(all_gold, all_pred, scheme=IOB2)
    report = classification_report(all_gold, all_pred, scheme=IOB2)
    return f1, report

# ===================
# Main (runner)
# ===================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="conll2003")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--text_column", default="tokens")
    ap.add_argument("--label_column", default="tags")
    ap.add_argument("--emb_dim", type=int, default=300, help="Model embedding size (should match FT dim unless projecting)")
    ap.add_argument("--proj_dim", type=int, default=None, help="Optional projection of embeddings before LSTM")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--output_dir", default="ner_word_bilstm_crf_out")
    # Vocab handling
    ap.add_argument("--vocab_file", default=None, help="Path to fixed vocab.txt (first two lines must be <pad>, <unk>)")
    ap.add_argument("--save_vocab", default=None, help="If set and building from train, save vocab to this path")
    ap.add_argument("--max_vocab", type=int, default=50000)
    ap.add_argument("--min_word_freq", type=int, default=1)
    # Normalization
    ap.add_argument("--lower", action="store_true")
    ap.add_argument("--normalize_digits", action="store_true")
    # FastText
    ap.add_argument("--use_fasttext", action="store_true")
    ap.add_argument("--fasttext_path", default=None)
    ap.add_argument("--fasttext_lock_frozen", action="store_true")
    ap.add_argument("--fasttext_oov", choices=["random","zero"], default="random")
    # Use explicit prebuilt embeddings path (optional future extension)
    args = ap.parse_args()

    # Load dataset
    raw = load_dataset(args.dataset, args.subset) if args.subset else load_dataset(args.dataset)
    assert "train" in raw, "Dataset must include a train split"
    assert "test" in raw, "Dataset must include a test split for final evaluation"
    train_split = raw["train"]
    dev_split = raw["validation"] if "validation" in raw else None
    test_split = raw["test"]

    # Labels from tags.lst
    tag_list, id2label, label2id = load_tags_list("tags.lst")

    # Vocabulary
    if args.vocab_file:
        word2id, id2word = load_vocab_file(args.vocab_file)
    else:
        word2id, id2word = build_word_vocab(
            raw, text_col=args.text_column,
            max_vocab=args.max_vocab, min_freq=args.min_word_freq,
            lower=args.lower, normalize_digits=args.normalize_digits
        )
        if args.save_vocab:
            os.makedirs(os.path.dirname(args.save_vocab) or ".", exist_ok=True)
            save_vocab_file(id2word, args.save_vocab)

    # Build tensor datasets
    Xtr, Ytr, Mtr = build_dataset(train_split, word2id, id2label, label2id,
                                  text_col=args.text_column, label_col=args.label_column,
                                  lower=args.lower, normalize_digits=args.normalize_digits)
    if dev_split is not None:
        Xdev, Ydev, Mdev = build_dataset(dev_split, word2id, id2label, label2id,
                                         text_col=args.text_column, label_col=args.label_column,
                                         lower=args.lower, normalize_digits=args.normalize_digits)
    Xte, Yte, Mte = build_dataset(test_split, word2id, id2label, label2id,
                                  text_col=args.text_column, label_col=args.label_column,
                                  lower=args.lower, normalize_digits=args.normalize_digits)

    # Batches
    train_batches = list(batchify(Xtr, Ytr, Mtr, args.batch_size))
    dev_batches = list(batchify(Xdev, Ydev, Mdev, args.batch_size)) if dev_split is not None else None
    test_batches = list(batchify(Xte, Yte, Mte, args.batch_size))

    # Training over seeds
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for seed in args.seeds:
        print(f"== Seed {seed} ==")
        set_seed(seed)

        # Prepare pretrained embeddings if requested
        pretrained_weight = None
        effective_emb_dim = args.emb_dim
        freeze_emb = False

        if args.use_fasttext:
            if not args.fasttext_path:
                print("Error: --use_fasttext requires --fasttext_path", file=sys.stderr)
                sys.exit(1)
            pretrained_weight, ft_dim = build_pretrained_weight_from_fasttext(
                id2word, args.fasttext_path, oov_policy=args.fasttext_oov
            )
            # Ensure embedding layer matches the FT dimension
            effective_emb_dim = ft_dim
            freeze_emb = args.fasttext_lock_frozen

        # Model
        model = WordBiLSTMCRF(
            num_words=len(word2id),
            num_tags=len(tag_list),
            emb_dim=effective_emb_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            proj_dim=args.proj_dim,  # if set and different, a projection layer will adapt to LSTM input
            padding_idx=word2id[PAD],
            pretrained_weight=pretrained_weight,
            freeze_emb=freeze_emb,
        ).to(device)

        # Optimizer
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_f1 = -1.0
        best_state = None

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_batches, opt, device)
            if dev_batches is not None:
                dev_f1, _ = evaluate(model, dev_batches, id2label, device)
                improved = dev_f1 > best_f1
                if improved:
                    best_f1 = dev_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"Epoch {epoch}: train_loss={train_loss:.4f} dev_f1={dev_f1:.4f} {'*' if improved else ''}")
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        # Always evaluate on test
        test_f1, report = evaluate(model, test_batches, id2label, device)
        results.append(test_f1)

        # Save artifacts
        outdir = os.path.join(args.output_dir, f"seed{seed}")
        os.makedirs(outdir, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "word2id": word2id, "label2id": label2id},
            os.path.join(outdir, "model.pt"),
        )
        with open(os.path.join(outdir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"dev_best_f1": float(best_f1), "test_f1": float(test_f1)}, f, indent=2)
        with open(os.path.join(outdir, "seqeval_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    mean_f1 = float(np.mean(results))
    std_f1 = float(np.std(results))
    with open(os.path.join(args.output_dir, "final.txt"), "w", encoding="utf-8") as of:
        of.write(json.dumps({"mean_f1": mean_f1, "std_f1": std_f1, "seeds": args.seeds}, indent=2) + "\n")
        of.write(f"LaTeX row: WordBiLSTM-CRF & {mean_f1:.2f} ± {std_f1:.2f}\n")
    print(f"Test F1: mean={mean_f1:.4f} std={std_f1:.4f} over {len(results)} seeds")

if __name__ == "__main__":
    main()
