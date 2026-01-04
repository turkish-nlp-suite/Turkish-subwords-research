#!/usr/bin/env python3
import argparse, os, json, random
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

# -----------------------
# Repro
# -----------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# -----------------------
# Constants
# -----------------------
PAD = "<pad>"
UNK = "<unk>"

# -----------------------
# IO: CoNLL-U
# -----------------------
def read_conllu(path: str):
    sents = []
    toks, upos, heads, rels, feats = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if toks:
                    sents.append({"tokens": toks, "upos": upos, "heads": heads, "rels": rels, "feats": feats})
                    toks, upos, heads, rels, feats = [], [], [], [], []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue  # skip multiword/empty nodes
            tok = cols[1]
            up = cols[3]
            feat = cols[5]
            head = int(cols[6])  # 0=root, otherwise 1-based word index
            rel = cols[7]
            toks.append(tok)
            upos.append(up)
            heads.append(head)
            rels.append(rel)
            feats.append(feat)
    if toks:
        sents.append({"tokens": toks, "upos": upos, "heads": heads, "rels": rels, "feats": feats})
    return sents

# -----------------------
# Morphology schema
# -----------------------
def parse_feats_str(feats_str: str) -> Dict[str, str]:
    if feats_str == "_" or not feats_str.strip():
        return {}
    out = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    return out

class MorphSchema:
    def __init__(self, attrs, values, value2id, id2value):
        self.attrs = attrs
        self.values = values
        self.value2id = value2id
        self.id2value = id2value

def build_schema(train_sents):
    attrs = set()
    values = {}
    for ex in train_sents:
        for fs in ex["feats"]:
            d = parse_feats_str(fs)
            for k, v in d.items():
                attrs.add(k)
                values.setdefault(k, set()).add(v)
    attrs = sorted(list(attrs))
    for a in attrs:
        values.setdefault(a, set())
        values[a].add("None")
        values[a] = sorted(list(values[a]))
    value2id = {a: {v: i for i, v in enumerate(values[a])} for a in attrs}
    id2value = {a: {i: v for i, v in enumerate(values[a])} for a in attrs}
    return MorphSchema(attrs, values, value2id, id2value)

# -----------------------
# Vocab + labels
# -----------------------
def load_vocab_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        seen = set()
        toks = []
        for line in f:
            w = line.strip()
            if not w:
                continue
            toks.append(w)
    itos = [PAD, UNK] + toks
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def build_word_vocab(train_sents, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for ex in train_sents:
        cnt.update(ex["tokens"])
    itos = [PAD, UNK]
    for w, c in cnt.items():
        if c >= min_freq and w not in itos:
            itos.append(w)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def build_label_maps(all_sents):
    upos = sorted({u for ex in all_sents for u in ex["upos"]})
    rels = sorted({r for ex in all_sents for r in ex["rels"]})
    upos2id = {u:i for i,u in enumerate(upos)}
    id2upos = {i:u for u,i in upos2id.items()}
    rel2id = {r:i for i,r in enumerate(rels)}
    id2rel = {i:r for r,i in rel2id.items()}
    return upos2id, id2upos, rel2id, id2rel

# -----------------------
# Split building (word-level; per-batch padding later)
# -----------------------
def build_split_word(sents, word2id, upos2id, rel2id, schema: MorphSchema):
    feats = []
    unk = word2id[UNK]
    for ex in sents:
        wids = [word2id.get(w, unk) for w in ex["tokens"]]
        pos = [upos2id[u] for u in ex["upos"]]
        rel = [rel2id[r] for r in ex["rels"]]
        head_word_idx = list(ex["heads"])  # list of ints, 0=root, else 1..W
        morph = {}
        for a in schema.attrs:
            vals = []
            for fs in ex["feats"]:
                d = parse_feats_str(fs)
                vals.append(schema.value2id[a].get(d.get(a, "None"), schema.value2id[a]["None"]))
            morph[a] = vals
        feats.append({
            "word_ids": torch.tensor(wids, dtype=torch.long),           # [W]
            "pos_labels": torch.tensor(pos, dtype=torch.long),          # [W]
            "head_word_idx": head_word_idx,                             # list[int] len W
            "rel_labels": torch.tensor(rel, dtype=torch.long),          # [W]
            "morph_labels": {a: torch.tensor(morph[a], dtype=torch.long) for a in schema.attrs},
        })
    return feats

# -----------------------
# Batching with padding
# -----------------------
def batchify_word(features, batch_size, pad_id):
    N = len(features)
    for i in range(0, N, batch_size):
        chunk = features[i:i+batch_size]
        seqs = [f["word_ids"] for f in chunk]
        maxlen = max(s.size(0) for s in seqs) if seqs else 0
        x = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        m = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
        pos_labels = []
        head_idx = []
        rel_labels = []
        morph_labels = []
        for r, f in enumerate(chunk):
            L = f["word_ids"].size(0)
            x[r, :L] = f["word_ids"]
            m[r, :L] = True
            pos_labels.append(f["pos_labels"])
            head_idx.append(f["head_word_idx"])
            rel_labels.append(f["rel_labels"])
            morph_labels.append(f["morph_labels"])
        yield {
            "x": x,           # [B,T]
            "mask": m,        # [B,T] bool
            "pos_labels": pos_labels,
            "head_word_idx": head_idx,
            "rel_labels": rel_labels,
            "morph_labels": morph_labels,
        }

# -----------------------
# FastText loading
# -----------------------
def load_fasttext_gensim(path: str):
    from gensim.models import KeyedVectors
    kv = KeyedVectors.load_word2vec_format(path, binary=path.endswith(".bin"))
    dim = kv.vector_size
    def getter(w):
        if w in kv:
            return kv[w]
        return None
    return getter, dim

def load_fasttext_official(path: str):
    import fasttext
    ft = fasttext.load_model(path)
    dim = ft.get_dimension()
    def getter(w):
        try:
            return ft.get_word_vector(w)
        except Exception:
            return None
    return getter, dim

def build_embedding_matrix(word2id, dim, getter=None, scale=0.02):
    W = np.random.normal(scale=scale, size=(len(word2id), dim)).astype("float32")
    # PAD as zeros; UNK random
    if word2id.get(PAD, None) is not None:
        W[word2id[PAD]] = 0.0
    if getter is not None:
        for w, i in word2id.items():
            if w == PAD:
                continue
            vec = getter(w)
            if vec is not None:
                W[i] = np.asarray(vec, dtype="float32")
    return torch.tensor(W)

# -----------------------
# Model components (word-level)
# -----------------------
class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.U = nn.Parameter(torch.zeros(in_dim + (1 if bias_x else 0),
                                          out_dim,
                                          in_dim + (1 if bias_y else 0)))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, x.new_ones(x.size(0), x.size(1), 1)], dim=-1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(y.size(0), y.size(1), 1)], dim=-1)
        xU = torch.einsum("bti, icj -> btcj", x, self.U)
        scores = torch.einsum("btcj, bkj -> btck", xU, y)  # [B,T,out_dim,K]
        return scores

class WordEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden=400, layers=2, dropout=0.3,
                 pad_idx=0, pretrained: Optional[torch.Tensor]=None, freeze_emb=False):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if pretrained is not None and pretrained.size(0) == vocab_size and pretrained.size(1) == emb_dim:
            self.emb.weight.data.copy_(pretrained)
        self.freeze_emb = freeze_emb
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden//2, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0, bidirectional=True
        )
    def forward(self, x, mask):
        e = self.emb(x)
        if self.freeze_emb:
            e = e.detach()
        e = self.dropout(e)
        h, _ = self.lstm(e)
        h = self.dropout(h)
        return h  # [B,T,H]

class PosDepMorphWordModel(nn.Module):
    def __init__(self, vocab_size, num_upos, num_rels, schema: MorphSchema,
                 emb_dim=300, enc_hidden=400, enc_layers=2, enc_dropout=0.3,
                 mlp_dim=400, mlp_dropout=0.33, morph_dropout=0.1,
                 pad_idx=0, pretrained_emb: Optional[torch.Tensor]=None, freeze_emb=False):
        super().__init__()
        self.encoder = WordEncoder(vocab_size, emb_dim, enc_hidden, enc_layers, enc_dropout,
                                   pad_idx=pad_idx, pretrained=pretrained_emb, freeze_emb=freeze_emb)
        H = enc_hidden
        self.pos_head = nn.Linear(H, num_upos)

        self.mlp_arc_dep = nn.Sequential(nn.Linear(H, mlp_dim), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.mlp_arc_head = nn.Sequential(nn.Linear(H, mlp_dim), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.mlp_rel_dep = nn.Sequential(nn.Linear(H, mlp_dim), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.mlp_rel_head = nn.Sequential(nn.Linear(H, mlp_dim), nn.ReLU(), nn.Dropout(mlp_dropout))
        self.arc_biaffine = Biaffine(mlp_dim, 1)
        self.rel_biaffine = Biaffine(mlp_dim, num_rels)
        self.root = nn.Parameter(torch.randn(1, 1, H) * 0.02)

        self.schema = schema
        self.morph_drop = nn.Dropout(morph_dropout)
        self.morph_clfs = nn.ModuleDict({a: nn.Linear(H, len(schema.values[a])) for a in schema.attrs})

    def forward(self, x, mask, pos_labels=None, head_word_idx=None, rel_labels=None, morph_labels=None):
        Hx = self.encoder(x, mask)  # [B,T,H]
        B, T, H = Hx.size()

        pos_logits_list, morph_logits_list, dep_arc_scores, rel_scores = [], [], [], []

        for b in range(B):
            L = int(mask[b].sum().item())
            if L == 0:
                # create dummies
                wr = Hx[b:b+1, :1, :]  # [1,1,H]
                wr = wr.squeeze(0)     # [1,H]
            else:
                wr = Hx[b, :L, :]      # [W,H]

            # POS
            pos_logits_list.append(self.pos_head(wr))  # [W,U]

            # Morph
            mlog = {}
            for a in self.schema.attrs:
                mlog[a] = self.morph_clfs[a](self.morph_drop(wr))  # [W,Va]
            morph_logits_list.append(mlog)

            # Dependency: add root to sentence representations
            root = self.root.expand(1, -1, -1).squeeze(0)  # [1,H]
            wr_rooted = torch.cat([root, wr], dim=0)       # [W+1,H]

            dep = self.mlp_arc_dep(wr_rooted)              # [W+1,M]
            head = self.mlp_arc_head(wr_rooted)            # [W+1,M]
            dep_b = dep.unsqueeze(0)                       # [1,W+1,M]
            head_b = head.unsqueeze(0)                     # [1,W+1,M]
            arc_b = self.arc_biaffine(dep_b, head_b)       # [1,W+1,1,W+1]
            arc = arc_b.squeeze(0).squeeze(1)              # [W+1,W+1]
            dep_arc_scores.append(arc[1:, :])              # [W, W+1] each dep chooses head in [0..W]

            dep_r = self.mlp_rel_dep(wr_rooted)            # [W+1,M]
            head_r = self.mlp_rel_head(wr_rooted)          # [W+1,M]
            dep_r_b = dep_r.unsqueeze(0)
            head_r_b = head_r.unsqueeze(0)
            rel_b = self.rel_biaffine(dep_r_b, head_r_b)   # [1,W+1,R,W+1]
            rel_scores.append(rel_b.squeeze(0))            # [W+1,R,W+1]

        losses = {}
        if pos_labels is not None:
            pl = []
            for b in range(B):
                y = pos_labels[b].to(pos_logits_list[b].device)    # [W]
                pl.append(nn.functional.cross_entropy(pos_logits_list[b], y))
            losses["pos"] = torch.stack(pl).mean()

        if head_word_idx is not None and rel_labels is not None:
            al, rl = [], []
            for b in range(B):
                arc = dep_arc_scores[b]  # [W, W+1]
                gold_heads = torch.tensor(head_word_idx[b], dtype=torch.long, device=arc.device)  # [W]
                al.append(nn.functional.cross_entropy(arc, gold_heads))
                rs = rel_scores[b]       # [W+1, R, W+1]
                dep_idx = torch.arange(gold_heads.size(0), device=arc.device)
                rel_logits = rs[dep_idx+1, :, gold_heads]   # [W,R]
                y_rel = rel_labels[b].to(rel_logits.device) # [W]
                rl.append(nn.functional.cross_entropy(rel_logits, y_rel))
            losses["arc"] = torch.stack(al).mean()
            losses["rel"] = torch.stack(rl).mean()

        if morph_labels is not None and len(self.schema.attrs) > 0:
            ml = []
            for b in range(B):
                for a in self.schema.attrs:
                    y = morph_labels[b][a].to(morph_logits_list[b][a].device)  # [W]
                    ml.append(nn.functional.cross_entropy(morph_logits_list[b][a], y))
            if ml:
                losses["morph"] = torch.stack(ml).mean()

        out = {
            "pos_logits": pos_logits_list,
            "arc_scores": dep_arc_scores,
            "rel_scores": rel_scores,
            "morph_logits": morph_logits_list,
        }
        if losses:
            out["losses"] = losses
            out["loss"] = losses.get("pos", 0.0) + losses.get("arc", 0.0) + losses.get("rel", 0.0) + losses.get("morph", 0.0)
        return out

# -----------------------
# Training and evaluation
# -----------------------
def train_one_epoch(model, batches, optimizer, device):
    model.train()
    tot = 0.0
    for batch in batches:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        out = model(
            x, m,
            pos_labels=batch["pos_labels"],
            head_word_idx=batch["head_word_idx"],
            rel_labels=batch["rel_labels"],
            morph_labels=batch["morph_labels"],
        )
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tot += loss.item()
    return tot / max(1, len(batches))

@torch.no_grad()
def evaluate(model, batches, id2upos, id2rel, schema: MorphSchema, device):
    model.eval()
    total_pos, correct_pos = 0, 0
    total_arcs, correct_uas, correct_las = 0, 0, 0
    morph_correct = {a: 0 for a in schema.attrs}
    morph_total = {a: 0 for a in schema.attrs}

    for batch in batches:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        out = model(x, m)
        # POS
        for logits, gold in zip(out["pos_logits"], batch["pos_labels"]):
            pred = logits.argmax(-1).cpu()
            gold = gold.cpu()
            correct_pos += (pred == gold).sum().item()
            total_pos += gold.numel()
        # UAS/LAS
        for arc_scores, rel_scores, heads_gold, rel_gold in zip(out["arc_scores"], out["rel_scores"], batch["head_word_idx"], batch["rel_labels"]):
            pred_heads = arc_scores.argmax(-1).cpu().tolist()  # [W], each in [0..W]
            gold_heads = heads_gold
            for i, (ph, gh) in enumerate(zip(pred_heads, gold_heads)):
                total_arcs += 1
                if ph == gh:
                    correct_uas += 1
                    rel_logits = rel_scores[i+1, :, ph]  # [R]
                    pred_rel = int(rel_logits.argmax(-1).item())
                    if pred_rel == int(rel_gold[i].item()):
                        correct_las += 1
        # Morph
        for morph_logits, gold_morph in zip(out["morph_logits"], batch["morph_labels"]):
            for a in schema.attrs:
                pred = morph_logits[a].argmax(-1).cpu()
                gold = gold_morph[a].cpu()
                morph_correct[a] += (pred == gold).sum().item()
                morph_total[a] += gold.numel()

    pos_acc = correct_pos / max(1, total_pos)
    uas = correct_uas / max(1, total_arcs)
    las = correct_las / max(1, total_arcs)
    metrics = {"pos_acc": pos_acc, "uas": uas, "las": las}
    if len(schema.attrs) > 0:
        per_attr = {f"morph_{a}_acc": (morph_correct[a] / morph_total[a] if morph_total[a] > 0 else 0.0)
                    for a in schema.attrs}
        micro_c = sum(morph_correct.values())
        micro_t = sum(morph_total.values())
        metrics.update(per_attr)
        metrics["morph_micro_acc"] = micro_c / max(1, micro_t)
    return metrics

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)

    # Vocab
    ap.add_argument("--vocab_file", type=str, default=None, help="Plain-text vocab file; <pad>/<unk> will be prepended.")
    ap.add_argument("--min_word_freq", type=int, default=1)

    # FastText
    ap.add_argument("--use_fasttext", action="store_true")
    ap.add_argument("--fasttext_path", type=str, default=None, help="Path to FastText vectors (.bin or .vec).")
    ap.add_argument("--fasttext_loader", type=str, choices=["gensim", "official"], default="gensim")
    ap.add_argument("--freeze_embeddings", action="store_true")

    # Model
    ap.add_argument("--emb_dim", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=400)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--enc_dropout", type=float, default=0.3)
    ap.add_argument("--mlp_dim", type=int, default=400)
    ap.add_argument("--mlp_dropout", type=float, default=0.33)
    ap.add_argument("--morph_dropout", type=float, default=0.1)

    # Train
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--output_dir", default="pos_dep_morph_word_out")
    args = ap.parse_args()

    train = read_conllu(args.train)
    dev = read_conllu(args.dev)
    test = read_conllu(args.test)

    upos2id, id2upos, rel2id, id2rel = build_label_maps(train + dev + test)
    schema = build_schema(train)

    if args.vocab_file:
        word2id, id2word = load_vocab_file(args.vocab_file)
    else:
        word2id, id2word = build_word_vocab(train, min_freq=args.min_word_freq)
    assert word2id[PAD] == 0 and word2id[UNK] == 1

    train_feats = build_split_word(train, word2id, upos2id, rel2id, schema)
    dev_feats = build_split_word(dev, word2id, upos2id, rel2id, schema)
    test_feats = build_split_word(test, word2id, upos2id, rel2id, schema)

    def make_batches(feats_list, bs, pad_id):
        return list(batchify_word(feats_list, bs, pad_id))

    train_batches = make_batches(train_feats, args.batch_size, pad_id=word2id[PAD])
    dev_batches   = make_batches(dev_feats, args.batch_size, pad_id=word2id[PAD])
    test_batches  = make_batches(test_feats, args.batch_size, pad_id=word2id[PAD])

    # Embeddings
    pretrained_tensor = None
    emb_dim = args.emb_dim
    if args.use_fasttext:
        assert args.fasttext_path, "--fasttext_path is required when --use_fasttext"
        if args.fasttext_loader == "gensim":
            getter, ft_dim = load_fasttext_gensim(args.fasttext_path)
        else:
            getter, ft_dim = load_fasttext_official(args.fasttext_path)
        emb_dim = ft_dim
        pretrained_tensor = build_embedding_matrix(word2id, dim=ft_dim, getter=getter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for seed in args.seeds:
        set_seed(seed)
        model = PosDepMorphWordModel(
            vocab_size=len(word2id),
            num_upos=len(upos2id),
            num_rels=len(rel2id),
            schema=schema,
            emb_dim=emb_dim,
            enc_hidden=args.hidden,
            enc_layers=args.layers,
            enc_dropout=args.enc_dropout,
            mlp_dim=args.mlp_dim,
            mlp_dropout=args.mlp_dropout,
            morph_dropout=args.morph_dropout,
            pad_idx=word2id[PAD],
            pretrained_emb=pretrained_tensor,
            freeze_emb=args.freeze_embeddings,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_dev = None
        best_state = None
        for ep in range(1, args.epochs + 1):
            tr_loss = train_one_epoch(model, train_batches, opt, device)
            dev_metrics = evaluate(model, dev_batches, id2upos, id2rel, schema, device)
            if best_dev is None or dev_metrics["las"] > best_dev["las"]:
                best_dev = dev_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {ep}: loss={tr_loss:.4f} dev POS={dev_metrics['pos_acc']:.4f} UAS={dev_metrics['uas']:.4f} LAS={dev_metrics['las']:.4f} morph_micro={dev_metrics.get('morph_micro_acc',0.0):.4f}")

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        test_metrics = evaluate(model, test_batches, id2upos, id2rel, schema, device)
        outdir = os.path.join(args.output_dir, f"seed{seed}")
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "dev_metrics.json"), "w") as f:
            json.dump(best_dev, f, indent=2)
        with open(os.path.join(outdir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "word2id": word2id,
                "upos2id": upos2id,
                "rel2id": rel2id,
                "schema": {"attrs": schema.attrs, "values": {a: schema.values[a] for a in schema.attrs}},
            },
            os.path.join(outdir, "model.pt"),
        )
        print("Test:", test_metrics)
        all_results.append(test_metrics)

    pos_accs = [r["pos_acc"] for r in all_results]
    uass = [r["uas"] for r in all_results]
    lass = [r["las"] for r in all_results]
    morph_micro = [r.get("morph_micro_acc", 0.0) for r in all_results]
    agg = {
        "pos_acc_mean": float(np.mean(pos_accs)),
        "pos_acc_std": float(np.std(pos_accs)),
        "uas_mean": float(np.mean(uass)),
        "uas_std": float(np.std(uass)),
        "las_mean": float(np.mean(lass)),
        "las_std": float(np.std(lass)),
        "morph_micro_acc_mean": float(np.mean(morph_micro)) if morph_micro else 0.0,
        "morph_micro_acc_std": float(np.std(morph_micro)) if morph_micro else 0.0,
        "seeds": args.seeds,
    }
    with open(os.path.join(args.output_dir, "final.txt"), "w") as f:
        f.write(json.dumps(agg, indent=2) + "\n")
        f.write(
            f"LaTeX row: WordBiLSTM + Biaffine & {agg['pos_acc_mean']:.2f}±{agg['pos_acc_std']:.2f} "
            f"& {agg['uas_mean']:.2f}±{agg['uas_std']:.2f} & {agg['las_mean']:.2f}±{agg['las_std']:.2f} "
            f"& {agg['morph_micro_acc_mean']:.2f}±{agg['morph_micro_acc_std']:.2f}\n"
        )

if __name__ == "__main__":
    main()
