#!/usr/bin/env python
#!/usr/bin/env python3
import argparse, os, json, random
from typing import List, Tuple, Dict
import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import f1_score as seqeval_f1, classification_report
from seqeval.scheme import IOB2

# -------------------
# Utils
# -------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_char_vocab(datasets, text_col="tokens", min_freq=1):
    from collections import Counter
    cnt = Counter()
    for ex in datasets["train"]:
        for tok in ex[text_col]:
            cnt.update(list(tok))
        cnt.update([" "])  # separator space
    itos = ["<pad>", "<unk>"]
    for ch, c in cnt.items():
        if c >= min_freq and ch not in itos:
            itos.append(ch)
    stoi = {c:i for i,c in enumerate(itos)}
    return stoi, itos

def tokens_to_char_seq(tokens: List[str], word_tags: List[str]) -> Tuple[List[str], List[str], List[Tuple[int,int]]]:
    chars, char_tags, spans = [], [], []
    cur = 0
    for idx, (tok, wt) in enumerate(zip(tokens, word_tags)):
        tchars = list(tok)
        if wt == "O":
            ttags = ["O"] * len(tchars)
        else:
            if wt.startswith("I-"):
                wt = "B-" + wt[2:]
            ent = wt[2:]
            ttags = [f"B-{ent}"] + [f"I-{ent}"] * max(0, len(tchars)-1)
        start = cur
        chars.extend(tchars); char_tags.extend(ttags); cur += len(tchars)
        end = cur
        spans.append((start, end))
        if idx != len(tokens) - 1:
            chars.append(" "); char_tags.append("O"); cur += 1
    return chars, char_tags, spans

def tags_to_ids(tags: List[str], label2id: Dict[str,int]) -> List[int]:
    return [label2id.get(t, label2id["O"]) for t in tags]

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

def decode_chars_to_word_tags(char_tag_ids: List[int], spans: List[Tuple[int,int]], id2label: Dict[int,str]) -> List[str]:
    out = []
    for s, e in spans:
        tag = id2label[char_tag_ids[s]] if 0 <= s < len(char_tag_ids) else "O"
        if tag.startswith("I-"):
            tag = "B-" + tag[2:]
        out.append(tag)
    return out

# -------------------
# Model: Char embedding + BiLSTM + CRF
# -------------------
class CRF(torch.nn.Module):
    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))
        torch.nn.init.uniform_(self.transitions, -0.1, 0.1)

    def neg_log_likelihood(self, emissions, tags, mask):
        # emissions: [B,T,C], tags: [B,T], mask: [B,T]
        log_den = self._compute_log_partition(emissions, mask)
        log_num = self._compute_joint_log_likelihood(emissions, tags, mask)
        return torch.mean(log_den - log_num)

    def _compute_joint_log_likelihood(self, emissions, tags, mask):
        B, T, C = emissions.size()
        score = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)  # [B,T]
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
        # Initialize log_alpha with first timestep; masked positions are ignored
        log_alpha = emissions[:, 0]  # [B,C]
        for t in range(1, T):
            emit_t = emissions[:, t].unsqueeze(1)  # [B,1,C]
            trans = self.transitions.unsqueeze(0)  # [1,C,C]
            scores = log_alpha.unsqueeze(2) + trans + emit_t  # [B,C,C]
            log_alpha_next = torch.logsumexp(scores, dim=1)   # [B,C]
            # If position t is masked (False), keep previous alpha (no update)
            log_alpha = torch.where(mask[:, t].unsqueeze(1), log_alpha_next, log_alpha)
        return torch.logsumexp(log_alpha, dim=1)

    def decode(self, emissions, mask):
        B, T, C = emissions.size()
        backpointers = []
        v = emissions[:, 0]  # [B,C]
        for t in range(1, T):
            emit_t = emissions[:, t].unsqueeze(1)           # [B,1,C]
            scores = v.unsqueeze(2) + self.transitions.unsqueeze(0) + emit_t  # [B,C,C]
            v, bp = torch.max(scores, dim=1)                # [B,C], [B,C]
            # If masked, do not update v and bp (keep previous)
            upd = mask[:, t].unsqueeze(1)
            v = torch.where(upd, v, emissions[:, t-1])  # keep something stable when masked
            backpointers.append(bp)
        best_last = torch.argmax(v, dim=1)  # [B]
        paths = []
        for b in range(B):
            seq = [best_last[b].item()]
            for t in reversed(backpointers):
                seq.append(t[b, seq[-1]].item())
            seq.reverse()
            L = int(mask[b].sum().item())
            paths.append(seq[:L])
        return paths

class CharBiLSTMCRF(torch.nn.Module):
    def __init__(self, num_chars, num_tags, emb_dim=128, hidden=256, num_layers=1, dropout=0.3):
        super().__init__()
        self.char_emb = torch.nn.Embedding(num_chars, emb_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(
            input_size=emb_dim, hidden_size=hidden//2, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0, bidirectional=True
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden, num_tags)
        self.crf = CRF(num_tags)
    def forward(self, x, mask, tags=None):
        emb = self.dropout(self.char_emb(x))            # [B,T,E]
        out, _ = self.lstm(emb)                         # [B,T,H]
        logits = self.fc(self.dropout(out))             # [B,T,C]
        if tags is not None:
            loss = self.crf.neg_log_likelihood(logits, tags, mask)
            return {"loss": loss, "logits": logits}
        else:
            pred = self.crf.decode(logits, mask)
            return {"pred_ids": pred, "logits": logits}

# -------------------
# Data prep
# -------------------
def build_dataset(split_ds, char2id, id2label, label2id, text_col="tokens", label_col="tags"):
    X, Y, M, spans_all = [], [], [], []
    for ex in split_ds:
        tokens = ex[text_col]
        word_tag_ids = ex[label_col]
        # ensure ints -> strings
        word_tags = [id2label[i] if isinstance(i, int) else str(i) for i in word_tag_ids]
        chars, char_tags, spans = tokens_to_char_seq(tokens, word_tags)
        char_ids = [char2id.get(c, char2id["<unk>"]) for c in chars]
        tag_ids = tags_to_ids(char_tags, label2id)
        X.append(char_ids); Y.append(tag_ids); spans_all.append(spans)
    x_pad, mask = pad_batch(X, pad=char2id["<pad>"])
    y_pad, _ = pad_batch(Y, pad=label2id["O"])
    return x_pad, y_pad, mask, spans_all

def batchify(x_pad, y_pad, mask, batch_size):
    N = x_pad.size(0)
    for i in range(0, N, batch_size):
        yield x_pad[i:i+batch_size], y_pad[i:i+batch_size], mask[i:i+batch_size]

def batchify_with_spans(x_pad, y_pad, mask, spans, batch_size):
    N = x_pad.size(0)
    for i in range(0, N, batch_size):
        yield (x_pad[i:i+batch_size], y_pad[i:i+batch_size], mask[i:i+batch_size]), spans[i:i+batch_size]

# -------------------
# Train/Eval
# -------------------
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
def evaluate(model, batched_with_spans, id2label, device):
    model.eval()
    all_pred_words, all_gold_words = [], []
    for (x, y, m), word_spans_batch in batched_with_spans:
        x = x.to(device); y = y.to(device); m = m.to(device)
        out = model(x, m, tags=None)
        preds = out["pred_ids"]  # list of lists (char-level tag ids)
        # build gold char ids trimmed to true lengths
        gold = [seq[:int(ms.sum().item())].tolist() for seq, ms in zip(y, m)]
        for p_char, g_char, word_spans in zip(preds, gold, word_spans_batch):
            pred_w = decode_chars_to_word_tags(p_char, word_spans, id2label)
            gold_w = decode_chars_to_word_tags(g_char, word_spans, id2label)
            all_pred_words.append(pred_w)
            all_gold_words.append(gold_w)
    f1 = seqeval_f1(all_gold_words, all_pred_words, scheme=IOB2)
    report = classification_report(all_gold_words, all_pred_words, scheme=IOB2)
    return f1, report

# -------------------
# Main
# -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="conll2003")
    ap.add_argument("--subset", default=None)
    ap.add_argument("--text_column", default="tokens")
    ap.add_argument("--label_column", default="tags")
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--output_dir", default="ner_char_bilstm_crf_out")
    ap.add_argument("--min_char_freq", type=int, default=1)
    args = ap.parse_args()

    # Load dataset
    raw = load_dataset(args.dataset, args.subset) if args.subset else load_dataset(args.dataset)

    # Labels from tags.lst to match your setup
    tag_list = [l.strip() for l in open("tags.lst","r").read().splitlines() if l.strip()]
    assert "O" in tag_list, "tags.lst must contain 'O'"
    id2label = {i:l for i,l in enumerate(tag_list)}
    label2id = {l:i for i,l in enumerate(tag_list)}

    # Build char vocab on train
    char2id, id2char = build_char_vocab(raw, text_col=args.text_column, min_freq=args.min_char_freq)

    # Tensor datasets
    Xtr, Ytr, Mtr, spans_tr = build_dataset(raw["train"], char2id, id2label, label2id,
                                            text_col=args.text_column, label_col=args.label_column)
    eval_split = "validation" if "validation" in raw else "test"
    Xdev, Ydev, Mdev, spans_dev = build_dataset(raw[eval_split], char2id, id2label, label2id,
                                                text_col=args.text_column, label_col=args.label_column)
    Xte, Yte, Mte, spans_te = build_dataset(raw["test"], char2id, id2label, label2id,
                                            text_col=args.text_column, label_col=args.label_column)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    results = []

    for seed in args.seeds:
        set_seed(seed)
        model = CharBiLSTMCRF(
            num_chars=len(char2id),
            num_tags=len(tag_list),
            emb_dim=args.emb_dim,
            hidden=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Batches
        train_batches = list(batchify(Xtr, Ytr, Mtr, args.batch_size))
        dev_batches = list(batchify_with_spans(Xdev, Ydev, Mdev, spans_dev, args.batch_size))
        test_batches = list(batchify_with_spans(Xte, Yte, Mte, spans_te, args.batch_size))

        best_f1 = -1.0
        best_state = None
        for epoch in range(1, args.epochs+1):
            train_loss = train_one_epoch(model, train_batches, opt, device)
            dev_f1, _ = evaluate(model, dev_batches, id2label, device)
            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {epoch}: train_loss={train_loss:.4f} dev_f1={dev_f1:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        test_f1, report = evaluate(model, test_batches, id2label, device)
        results.append(test_f1)

        outdir = os.path.join(args.output_dir, f"seed{seed}")
        os.makedirs(outdir, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "char2id": char2id, "label2id": label2id},
                   os.path.join(outdir, "model.pt"))
        with open(os.path.join(outdir, "metrics.json"), "w") as f:
            json.dump({"dev_best_f1": float(best_f1), "test_f1": float(test_f1)}, f, indent=2)
        with open(os.path.join(outdir, "seqeval_report.txt"), "w") as f:
            f.write(report)

    mean_f1 = float(np.mean(results))
    std_f1 = float(np.std(results))
    with open(os.path.join(args.output_dir, "final.txt"), "w") as of:
        of.write(json.dumps({"mean_f1": mean_f1, "std_f1": std_f1, "seeds": args.seeds}, indent=2) + "\n")
        of.write(f"LaTeX row: CharBiLSTM-CRF & {mean_f1:.2f} ± {std_f1:.2f}\n")

if __name__ == "__main__":
    main()
