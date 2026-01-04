#!/usr/bin/env python3
# ===== File: pos_dep_morph_word.py =====
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
# Subword cache IO
# -----------------------
def load_json(path: Optional[str]) -> Optional[dict]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fetch_subwords_from_cache(token: str, cache: Optional[dict], lower=False):
    key = token.lower() if lower else token
    if cache is None:
        return [token]
    sw = cache.get(key)
    if sw is None or not isinstance(sw, list) or not sw:
        return [token]
    return sw

# -----------------------
# Vocab + labels
# -----------------------
def load_vocab_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        toks = []
        for line in f:
            w = line.strip()
            if not w:
                continue
            toks.append(w)
    itos = [PAD, UNK] + toks
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def load_subword_vocab_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
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
# Split building (subword-level)
# -----------------------
def build_split_subword(
    sents, subword2id, upos2id, rel2id, schema: MorphSchema, cache: Optional[dict],
    lower: bool = False
):
    feats = []
    unk = subword2id[UNK]

    for ex in sents:
        words = ex["tokens"]
        upos = ex["upos"]
        heads_w = ex["heads"]   # word-level: 0..W, 0=root
        rels = ex["rels"]
        feats_strs = ex["feats"]

        sw_lists = [fetch_subwords_from_cache(w, cache, lower=lower) for w in words]

        first_sw_idx = []
        flat_sw = []
        for sws in sw_lists:
            first_sw_idx.append(len(flat_sw))
            flat_sw.extend(sws)

        sw_ids = [subword2id.get(sw, unk) for sw in flat_sw]
        T = len(flat_sw)

        # Edge case: sentence with zero subwords (shouldn't happen, but guard)
        if T == 0:
            feats.append({
                "subword_ids": torch.tensor([], dtype=torch.long),
                "pos_labels_sw": torch.tensor([], dtype=torch.long),
                "head_sw_idx": [],
                "rel_labels_sw": torch.tensor([], dtype=torch.long),
                "morph_labels_sw": {a: torch.tensor([], dtype=torch.long) for a in schema.attrs},
            })
            continue

        pos_labels_sw = torch.full((T,), -100, dtype=torch.long)  # use ignore_index=-100
        morph_labels_sw = {a: torch.full((T,), -100, dtype=torch.long) for a in schema.attrs}

        for wi, (u, fs) in enumerate(zip(upos, feats_strs)):
            sidx = first_sw_idx[wi]
            pos_labels_sw[sidx] = upos2id[u]
            d = parse_feats_str(fs)
            for a in schema.attrs:
                vid = schema.value2id[a].get(d.get(a, "None"), schema.value2id[a]["None"])
                morph_labels_sw[a][sidx] = vid

        # Dependency at subword level: supervise only first subword per word
        head_sw_idx = torch.full((T,), -1, dtype=torch.long)  # -1 = ignore in loss
        rel_labels_sw = torch.full((T,), -100, dtype=torch.long)

        for wi, (h_w, r) in enumerate(zip(heads_w, rels)):
            dep_first = first_sw_idx[wi]
            if h_w == 0:
                head_sw = 0  # root
            else:
                head_first = first_sw_idx[h_w - 1]
                head_sw = head_first + 1  # shift by 1 to reserve 0 for root
            head_sw_idx[dep_first] = head_sw
            rel_labels_sw[dep_first] = rel2id[r]

        feats.append({
            "subword_ids": torch.tensor(sw_ids, dtype=torch.long),     # [T]
            "pos_labels_sw": pos_labels_sw,                             # [T] with -100 ignored
            "head_sw_idx": head_sw_idx.tolist(),                        # list[int] len T, -1 ignored rows; values in [0..T]
            "rel_labels_sw": rel_labels_sw,                             # [T] with -100 ignored
            "morph_labels_sw": {a: morph_labels_sw[a] for a in schema.attrs},
        })
    return feats

# -----------------------
# Batching with padding: word
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
            if L > 0:
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
# Batching with padding: subword
# -----------------------
def batchify_subword(features, batch_size, pad_id):
    N = len(features)
    for i in range(0, N, batch_size):
        chunk = features[i:i+batch_size]
        seqs = [f["subword_ids"] for f in chunk]
        maxlen = max((s.size(0) for s in seqs), default=0)
        x = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        m = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
        pos_labels_sw = []
        head_sw_idx = []
        rel_labels_sw = []
        morph_labels_sw = []
        for r, f in enumerate(chunk):
            L = f["subword_ids"].size(0)
            if L > 0:
                x[r, :L] = f["subword_ids"]
                m[r, :L] = True
            pos_labels_sw.append(f["pos_labels_sw"])
            head_sw_idx.append(f["head_sw_idx"])
            rel_labels_sw.append(f["rel_labels_sw"])
            morph_labels_sw.append(f["morph_labels_sw"])
        yield {
            "x": x,                 # [B,T]
            "mask": m,              # [B,T]
            "pos_labels_sw": pos_labels_sw,     # list[Tensor[T]]
            "head_sw_idx": head_sw_idx,         # list[list[int]]
            "rel_labels_sw": rel_labels_sw,     # list[Tensor[T]]
            "morph_labels_sw": morph_labels_sw, # list[dict[str, Tensor[T]]]
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
# Model components (shared)
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

# -----------------------
# Word-level model
# -----------------------
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
                wr = Hx[b:b+1, :1, :]
                wr = wr.squeeze(0)
            else:
                wr = Hx[b, :L, :]

            pos_logits_list.append(self.pos_head(wr))

            mlog = {}
            for a in self.schema.attrs:
                mlog[a] = self.morph_clfs[a](self.morph_drop(wr))
            morph_logits_list.append(mlog)

            root = self.root.expand(1, -1, -1).squeeze(0)
            wr_rooted = torch.cat([root, wr], dim=0)

            dep = self.mlp_arc_dep(wr_rooted)
            head = self.mlp_arc_head(wr_rooted)
            dep_b = dep.unsqueeze(0)
            head_b = head.unsqueeze(0)
            arc_b = self.arc_biaffine(dep_b, head_b)
            arc = arc_b.squeeze(0).squeeze(1)
            dep_arc_scores.append(arc[1:, :])  # [W, W+1]

            dep_r = self.mlp_rel_dep(wr_rooted)
            head_r = self.mlp_rel_head(wr_rooted)
            dep_r_b = dep_r.unsqueeze(0)
            head_r_b = head_r.unsqueeze(0)
            rel_b = self.rel_biaffine(dep_r_b, head_r_b)
            rel_scores.append(rel_b.squeeze(0))  # [W+1,R,W+1]

        losses = {}
        if pos_labels is not None:
            pl = []
            for b in range(B):
                y = pos_labels[b].to(pos_logits_list[b].device)
                pl.append(nn.functional.cross_entropy(pos_logits_list[b], y))
            losses["pos"] = torch.stack(pl).mean()

        if head_word_idx is not None and rel_labels is not None:
            al, rl = [], []
            for b in range(B):
                arc = dep_arc_scores[b]  # [W, W+1]
                if arc.numel() == 0:
                    continue
                gold_heads = torch.tensor(head_word_idx[b], dtype=torch.long, device=arc.device)
                al.append(nn.functional.cross_entropy(arc, gold_heads))
                rs = rel_scores[b]       # [W+1, R, W+1]
                dep_idx = torch.arange(gold_heads.size(0), device=arc.device)
                rel_logits = rs[dep_idx+1, :, gold_heads]   # [W,R]
                y_rel = rel_labels[b].to(rel_logits.device)
                rl.append(nn.functional.cross_entropy(rel_logits, y_rel))
            if al:
                losses["arc"] = torch.stack(al).mean()
            if rl:
                losses["rel"] = torch.stack(rl).mean()

        if morph_labels is not None and len(self.schema.attrs) > 0:
            ml = []
            for b in range(B):
                for a in self.schema.attrs:
                    y = morph_labels[b][a].to(morph_logits_list[b][a].device)
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
# Subword-level model
# -----------------------
class PosDepMorphSubwordModel(nn.Module):
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

    def forward(self, x, mask,
                pos_labels_sw=None, head_sw_idx=None, rel_labels_sw=None, morph_labels_sw=None):
        Hx = self.encoder(x, mask)  # [B,T,H]
        B, T, H = Hx.size()

        pos_logits_list, morph_logits_list, dep_arc_scores, rel_scores = [], [], [], []

        for b in range(B):
            L = int(mask[b].sum().item())
            if L == 0:
                # Avoid zero-length; create a dummy single step so heads range [0..1]
                sw = Hx[b:b+1, :1, :].squeeze(0)
            else:
                sw = Hx[b, :L, :]  # [T_b, H]

            pos_logits_list.append(self.pos_head(sw))

            mlog = {}
            for a in self.schema.attrs:
                mlog[a] = self.morph_clfs[a](self.morph_drop(sw))
            morph_logits_list.append(mlog)

            root = self.root.expand(1, -1, -1).squeeze(0)  # [1,H]
            sw_rooted = torch.cat([root, sw], dim=0)       # [T_b+1, H]

            dep = self.mlp_arc_dep(sw_rooted)
            head = self.mlp_arc_head(sw_rooted)
            arc_b = self.arc_biaffine(dep.unsqueeze(0), head.unsqueeze(0)).squeeze(0).squeeze(0)  # [T_b+1, T_b+1]
            dep_arc_scores.append(arc_b[1:, :])  # [T_b, T_b+1], each subword chooses head in [0..T_b]

            dep_r = self.mlp_rel_dep(sw_rooted)
            head_r = self.mlp_rel_head(sw_rooted)
            rel_b = self.rel_biaffine(dep_r.unsqueeze(0), head_r.unsqueeze(0)).squeeze(0)  # [T_b+1, R, T_b+1]
            rel_scores.append(rel_b)

        losses = {}
        if pos_labels_sw is not None:
            pl = []
            for b in range(B):
                logits = pos_logits_list[b]
                y = pos_labels_sw[b].to(logits.device)
                y = y[:logits.size(0)]
                pl.append(nn.functional.cross_entropy(logits, y, ignore_index=-100))
            losses["pos"] = torch.stack(pl).mean()

        if morph_labels_sw is not None and len(self.schema.attrs) > 0:
            ml = []
            for b in range(B):
                for a in self.schema.attrs:
                    logits = morph_logits_list[b][a]
                    y = morph_labels_sw[b][a].to(logits.device)[:logits.size(0)]
                    ml.append(nn.functional.cross_entropy(logits, y, ignore_index=-100))
            if ml:
                losses["morph"] = torch.stack(ml).mean()

        if head_sw_idx is not None and rel_labels_sw is not None:
            al, rl = [], []
            for b in range(B):
                arc = dep_arc_scores[b]  # [T_b, T_b+1]
                Lb = arc.size(0)
                if Lb == 0:
                    continue
                # FIX: slice gold and rel labels to Lb, ensure correct dtype
                gold_heads = torch.tensor(head_sw_idx[b], dtype=torch.long, device=arc.device)[:Lb]
                rel_gold = rel_labels_sw[b].to(arc.device)[:Lb].long()

                # FIX: compute keep from the sliced gold_heads, align with arc
                keep = (gold_heads >= 0).nonzero(as_tuple=False).squeeze(-1)
                if keep.numel() > 0:
                    arc_sel = arc.index_select(0, keep)          # [N, T_b+1]
                    gh = gold_heads.index_select(0, keep)         # [N], long


                    arc_sel = arc_sel.squeeze(1)

                    # cross_entropy expects (N,C) logits and (N,) long targets
                    al.append(nn.functional.cross_entropy(arc_sel, gh))

                    rs = rel_scores[b]                             # [T_b+1, R, T_b+1]
                    dep_idx = (keep + 1)                           # shift due to root
                    # Gather relation logits for each (dep, head)
                    rel_logits = rs[dep_idx, :, gh]                # [N, R]
                    y_rel = rel_gold.index_select(0, keep)         # [N]
                    rl.append(nn.functional.cross_entropy(rel_logits, y_rel))
            if al:
                losses["arc"] = torch.stack(al).mean()
            if rl:
                losses["rel"] = torch.stack(rl).mean()

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
def train_one_epoch_word(model, batches, optimizer, device):
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
def evaluate_word(model, batches, id2upos, id2rel, schema: MorphSchema, device):
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
            if arc_scores.numel() == 0:
                continue
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

def train_one_epoch_subword(model, batches, optimizer, device):
    model.train()
    tot = 0.0
    for batch in batches:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        out = model(
            x, m,
            pos_labels_sw=batch["pos_labels_sw"],
            head_sw_idx=batch["head_sw_idx"],
            rel_labels_sw=batch["rel_labels_sw"],
            morph_labels_sw=batch["morph_labels_sw"],
        )
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tot += loss.item()
    return tot / max(1, len(batches))

@torch.no_grad()
def evaluate_subword(model, batches, id2upos, id2rel, schema: MorphSchema, device):
    model.eval()
    total_pos, correct_pos = 0, 0
    total_arcs, correct_uas, correct_las = 0, 0, 0
    morph_correct = {a: 0 for a in schema.attrs}
    morph_total = {a: 0 for a in schema.attrs}

    for batch in batches:
        x = batch["x"].to(device)
        m = batch["mask"].to(device)
        out = model(x, m)

        # POS and Morph evaluated only on first-subword positions, encoded by labels != -100
        for logits, gold in zip(out["pos_logits"], batch["pos_labels_sw"]):
            L = logits.size(0)
            g = gold[:L].cpu()
            p = logits[:L].argmax(-1).cpu()
            mask_valid = g != -100
            correct_pos += (p[mask_valid] == g[mask_valid]).sum().item()
            total_pos += mask_valid.sum().item()

        # UAS/LAS on first subwords only
        for arc_scores, rel_scores, heads_gold, rel_gold in zip(out["arc_scores"], out["rel_scores"], batch["head_sw_idx"], batch["rel_labels_sw"]):
            if arc_scores.numel() == 0:
                continue
            gold_heads = torch.tensor(heads_gold, dtype=torch.long)
            L = arc_scores.size(0)
            keep = (gold_heads[:L] >= 0).nonzero(as_tuple=False).squeeze(-1)
            if keep.numel() == 0:
                continue
            pred_heads = arc_scores.argmax(-1).cpu()  # [L] each in [0..L]
            for idx in keep.tolist():
                ph = int(pred_heads[idx].item())
                gh = int(gold_heads[idx].item())
                total_arcs += 1
                if ph == gh:
                    correct_uas += 1
                    rel_logits = rel_scores[idx+1, :, ph]  # [R]
                    pred_rel = int(rel_logits.argmax(-1).item())
                    if pred_rel == int(rel_gold[idx].item()):
                        correct_las += 1

        # Morph
        for morph_logits, gold_morph in zip(out["morph_logits"], batch["morph_labels_sw"]):
            for a in schema.attrs:
                logits = morph_logits[a]
                L = logits.size(0)
                g = gold_morph[a][:L].cpu()
                p = logits[:L].argmax(-1).cpu()
                mask_valid = g != -100
                morph_correct[a] += (p[mask_valid] == g[mask_valid]).sum().item()
                morph_total[a] += mask_valid.sum().item()

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

    # Mode
    ap.add_argument("--use_subwords", action="store_true", help="Enable subword mode with cache-expanded inputs")

    # Vocab
    ap.add_argument("--vocab_file", type=str, default=None, help="Plain-text vocab file; <pad>/<unk> will be prepended. For subword mode, this is subword vocab.")
    ap.add_argument("--min_word_freq", type=int, default=1)

    # Subword cache
    ap.add_argument("--subword_cache", type=str, default=None, help="Fallback subword cache JSON for all splits")
    ap.add_argument("--subword_cache_train", type=str, default=None)
    ap.add_argument("--subword_cache_dev", type=str, default=None)
    ap.add_argument("--subword_cache_test", type=str, default=None)
    ap.add_argument("--subword_lower", action="store_true", help="Lowercase tokens before cache lookup")

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
    ap.add_argument("--output_dir", default="pos_dep_morph_out")
    args = ap.parse_args()

    train = read_conllu(args.train)
    dev = read_conllu(args.dev)
    test = read_conllu(args.test)

    upos2id, id2upos, rel2id, id2rel = build_label_maps(train + dev + test)
    schema = build_schema(train)

    if args.use_subwords:
        if args.vocab_file:
            sub2id, id2sub = load_subword_vocab_file(args.vocab_file)
        else:
            sub2id, id2sub = build_word_vocab(train, min_freq=args.min_word_freq)
        assert sub2id[PAD] == 0 and sub2id[UNK] == 1

        cache_train = load_json(args.subword_cache_train) or load_json(args.subword_cache)
        cache_dev   = load_json(args.subword_cache_dev)   or load_json(args.subword_cache)
        cache_test  = load_json(args.subword_cache_test)  or load_json(args.subword_cache)

        train_feats = build_split_subword(train, sub2id, upos2id, rel2id, schema, cache_train, lower=args.subword_lower)
        dev_feats   = build_split_subword(dev,   sub2id, upos2id, rel2id, schema, cache_dev,   lower=args.subword_lower)
        test_feats  = build_split_subword(test,  sub2id, upos2id, rel2id, schema, cache_test,  lower=args.subword_lower)

        def make_batches_sw(feats_list, bs, pad_id):
            return list(batchify_subword(feats_list, bs, pad_id))
        train_batches = make_batches_sw(train_feats, args.batch_size, pad_id=sub2id[PAD])
        dev_batches   = make_batches_sw(dev_feats,   args.batch_size, pad_id=sub2id[PAD])
        test_batches  = make_batches_sw(test_feats,  args.batch_size, pad_id=sub2id[PAD])

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
            pretrained_tensor = build_embedding_matrix(sub2id, dim=ft_dim, getter=getter)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(args.output_dir, exist_ok=True)
        all_results = []

        for seed in args.seeds:
            set_seed(seed)
            model = PosDepMorphSubwordModel(
                vocab_size=len(sub2id),
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
                pad_idx=sub2id[PAD],
                pretrained_emb=pretrained_tensor,
                freeze_emb=args.freeze_embeddings,
            ).to(device)

            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_dev = None
            best_state = None
            for ep in range(1, args.epochs + 1):
                tr_loss = train_one_epoch_subword(model, train_batches, opt, device)
                dev_metrics = evaluate_subword(model, dev_batches, id2upos, id2rel, schema, device)
                if best_dev is None or dev_metrics["las"] > best_dev["las"]:
                    best_dev = dev_metrics
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    morph_micro = dev_metrics.get('morph_micro_acc') or 0.0
                print(f"Epoch {ep}: loss={tr_loss:.4f} dev POS={dev_metrics['pos_acc']:.4f} UAS={dev_metrics['uas']:.4f} LAS={dev_metrics['las']:.4f} morph_micro={dev_metrics.get('morph_micro_acc',0.0):.4f}")

            if best_state is not None:
                model.load_state_dict(best_state, strict=True)

            test_metrics = evaluate_subword(model, test_batches, id2upos, id2rel, schema, device)
            outdir = os.path.join(args.output_dir, f"seed{seed}")
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, "dev_metrics.json"), "w") as f:
                json.dump(best_dev, f, indent=2)
            with open(os.path.join(outdir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "subword2id": sub2id,
                    "upos2id": upos2id,
                    "rel2id": rel2id,
                    "schema": {"attrs": schema.attrs, "values": {a: schema.values[a] for a in schema.attrs}},
                    "mode": "subword",
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
                f"LaTeX row: SubwordBiLSTM + Biaffine & {agg['pos_acc_mean']:.2f}±{agg['pos_acc_std']:.2f} "
                f"& {agg['uas_mean']:.2f}±{agg['uas_std']:.2f} & {agg['las_mean']:.2f}±{agg['las_std']:.2f} "
                f"& {agg['morph_micro_acc_mean']:.2f}±{agg['morph_micro_acc_std']:.2f}\n"
            )
        return

    # Word mode (default)
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
            tr_loss = train_one_epoch_word(model, train_batches, opt, device)
            dev_metrics = evaluate_word(model, dev_batches, id2upos, id2rel, schema, device)
            if best_dev is None or dev_metrics["las"] > best_dev["las"]:
                best_dev = dev_metrics
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"Epoch {ep}: loss={tr_loss:.4f} dev POS={dev_metrics['pos_acc']:.4f} UAS={dev_metrics['uas']:.4f} LAS={dev_metrics['las']:.4f} morph_micro={dev_metrics.get('morph_micro_acc',0.0):.4f}")

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        test_metrics = evaluate_word(model, test_batches, id2upos, id2rel, schema, device)
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
                "mode": "word",
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
