#!/usr/bin/env python3
#!/usr/bin/env python3
import argparse, os, json, random
from typing import List, Dict, Tuple
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
# Character stream + word spans
# -----------------------
def tokens_to_char_stream(tokens: List[str], sep=" ") -> Tuple[List[str], List[Tuple[int,int]]]:
    chars = []
    spans = []
    cur = 0
    for i, tok in enumerate(tokens):
        start = cur
        for c in tok:
            chars.append(c)
            cur += 1
        end = cur
        spans.append((start, end))
        if i != len(tokens) - 1:
            chars.append(sep)
            cur += 1
    return chars, spans

# -----------------------
# Vocab + labels
# -----------------------
def build_char_vocab(train_sents, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for ex in train_sents:
        for tok in ex["tokens"]:
            cnt.update(tok)
        cnt.update([" "])  # ensure space is present
    itos = ["<pad>", "<unk>"]
    for ch, c in cnt.items():
        if c >= min_freq and ch not in itos:
            itos.append(ch)
    stoi = {c:i for i,c in enumerate(itos)}
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
# Build split features (store variable-length tensors; pad per-batch)
# -----------------------
def build_split(sents, char2id, upos2id, rel2id, schema: MorphSchema):
    feats = []
    for ex in sents:
        tokens = ex["tokens"]
        chars, spans = tokens_to_char_stream(tokens, sep=" ")
        char_ids = [char2id.get(c, char2id["<unk>"]) for c in chars]
        char_ids_t = torch.tensor(char_ids, dtype=torch.long)
        char_mask_t = torch.ones(len(char_ids), dtype=torch.bool)

        pos = torch.tensor([upos2id[u] for u in ex["upos"]], dtype=torch.long)   # [W]
        rel = torch.tensor([rel2id[r] for r in ex["rels"]], dtype=torch.long)    # [W]
        head_word_idx = list(ex["heads"])  # list of ints, 0=root, else 1..W

        morph = {}
        for a in schema.attrs:
            vals = []
            for fs in ex["feats"]:
                d = parse_feats_str(fs)
                vals.append(schema.value2id[a].get(d.get(a, "None"), schema.value2id[a]["None"]))
            morph[a] = torch.tensor(vals, dtype=torch.long)  # [W]

        feats.append({
            "char_ids": char_ids_t,          # 1D LongTensor (var len)
            "char_mask": char_mask_t,        # 1D BoolTensor (var len)
            "word_spans": spans,             # list[(start,end)]
            "pos_labels": pos,               # [W]
            "head_word_idx": head_word_idx,  # list[int] length W
            "rel_labels": rel,               # [W]
            "morph_labels": morph,           # dict a -> [W]
        })
    return feats

# -----------------------
# Batching with per-batch padding (fixes stack size mismatch)
# -----------------------
def batchify_with_spans(features, batch_size, pad_id):
    N = len(features)
    for i in range(0, N, batch_size):
        chunk = features[i:i+batch_size]

        # Per-batch pad char_ids/mask
        seqs = [f["char_ids"] for f in chunk]
        masks = [f["char_mask"] for f in chunk]
        maxlen = max(s.size(0) for s in seqs) if seqs else 0
        x = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
        m = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
        for r, (s, ms) in enumerate(zip(seqs, masks)):
            L = s.size(0)
            x[r, :L] = s
            m[r, :L] = ms

        pos_labels = [f["pos_labels"] for f in chunk]         # list[LongTensor W]
        head_idx   = [f["head_word_idx"] for f in chunk]      # list[list int]
        rel_labels = [f["rel_labels"] for f in chunk]         # list[LongTensor W]
        spans      = [f["word_spans"] for f in chunk]         # list[list[(s,e)]]
        morph_labels = [f["morph_labels"] for f in chunk]     # list[dict a->LongTensor W]

        yield {
            "x": x, "mask": m,
            "pos_labels": pos_labels,
            "head_word_idx": head_idx,
            "rel_labels": rel_labels,
            "word_spans": spans,
            "morph_labels": morph_labels,
        }

# -----------------------
# Model components
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

class CharEncoder(nn.Module):
    def __init__(self, num_chars, emb_dim=128, hidden=256, layers=1, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(num_chars, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hidden//2, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0, bidirectional=True
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        e = self.drop(self.emb(x))         # [B,T,E]
        h, _ = self.lstm(e)                # [B,T,H]
        h = self.drop(h)
        return h

class PosDepMorphCharModel(nn.Module):
    def __init__(self, num_chars, num_upos, num_rels, schema: MorphSchema,
                 emb_dim=128, enc_hidden=256, enc_layers=1, enc_dropout=0.3,
                 mlp_dim=400, mlp_dropout=0.33, morph_dropout=0.1):
        super().__init__()
        self.encoder = CharEncoder(num_chars, emb_dim, enc_hidden, enc_layers, enc_dropout)
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

    def words_from_chars(self, Hc, word_spans):
        # Hc: [B,T,H]; word_spans: list of list[(s,e)]
        B, T, H = Hc.size()
        reps = []
        for b in range(B):
            sents = word_spans[b]
            if len(sents) == 0:
                reps.append(Hc[b, :1])  # dummy
                continue
            wrs = []
            for (s, e) in sents:
                s = min(max(s, 0), T-1)
                wrs.append(Hc[b, s])  # first char rep
            reps.append(torch.stack(wrs, dim=0))  # [W,H]
        return reps

    def forward(self, x, mask, word_spans, pos_labels=None, head_word_idx=None, rel_labels=None, morph_labels=None):
        Hc = self.encoder(x, mask)  # [B,T,H]
        B, T, H = Hc.size()

        word_reps = self.words_from_chars(Hc, word_spans)  # list of [W,H]
        pos_logits_list, morph_logits_list = [], []

        for b in range(B):
            wr = word_reps[b]                         # [W,H]
            pos_logits_list.append(self.pos_head(wr)) # [W,U]
            mlog = {}
            for a in self.schema.attrs:
                mlog[a] = self.morph_clfs[a](self.morph_drop(wr))  # [W,Va]
            morph_logits_list.append(mlog)

        dep_arc_scores, rel_scores = [], []
        for b in range(B):
            wr = word_reps[b]                               # [W,H]
            root = self.root.expand(1, -1, -1).squeeze(0)   # [1,H]
            wr_rooted = torch.cat([root, wr], dim=0)        # [W+1,H]
            # Project
            dep = self.mlp_arc_dep(wr_rooted)               # [W+1,M]
            head = self.mlp_arc_head(wr_rooted)             # [W+1,M]

            # Add batch dim for biaffine: [1, W+1, M]
            dep_b = dep.unsqueeze(0)
            head_b = head.unsqueeze(0)

            # Arc scores: output [1, W+1, 1, W+1] -> squeeze to [W+1, W+1]
            arc_b = self.arc_biaffine(dep_b, head_b)        # [1, W+1, 1, W+1]
            arc = arc_b.squeeze(0).squeeze(1)               # [W+1, W+1]
            dep_arc_scores.append(arc[1:, :])               # [W, W+1]

            # Relations
            dep_r = self.mlp_rel_dep(wr_rooted)             # [W+1,M]
            head_r = self.mlp_rel_head(wr_rooted)           # [W+1,M]
            dep_r_b = dep_r.unsqueeze(0)                    # [1, W+1, M]
            head_r_b = head_r.unsqueeze(0)                  # [1, W+1, M]

            # rel_b: [1, W+1, R, W+1] -> squeeze to [W+1, R, W+1]
            rel_b = self.rel_biaffine(dep_r_b, head_r_b)    # [1, W+1, R, W+1]
            rel = rel_b.squeeze(0)                          # [W+1, R, W+1]
            rel_scores.append(rel)

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
            x, m, word_spans=batch["word_spans"],
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
        out = model(x, m, word_spans=batch["word_spans"])
        # POS
        for logits, gold in zip(out["pos_logits"], batch["pos_labels"]):
            pred = logits.argmax(-1).cpu()
            gold = gold.cpu()
            correct_pos += (pred == gold).sum().item()
            total_pos += gold.numel()
        # UAS/LAS
        for arc_scores, rel_scores, heads_gold, rel_gold in zip(out["arc_scores"], out["rel_scores"], batch["head_word_idx"], batch["rel_labels"]):
            pred_heads = arc_scores.argmax(-1).cpu().tolist()  # [W]
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
    ap.add_argument("--emb_dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--enc_dropout", type=float, default=0.3)
    ap.add_argument("--mlp_dim", type=int, default=400)
    ap.add_argument("--mlp_dropout", type=float, default=0.33)
    ap.add_argument("--morph_dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--output_dir", default="pos_dep_morph_char_out")
    ap.add_argument("--min_char_freq", type=int, default=1)
    args = ap.parse_args()

    train = read_conllu(args.train)
    dev = read_conllu(args.dev)
    test = read_conllu(args.test)

    upos2id, id2upos, rel2id, id2rel = build_label_maps(train + dev + test)
    schema = build_schema(train)
    char2id, id2char = build_char_vocab(train, min_freq=args.min_char_freq)

    train_feats = build_split(train, char2id, upos2id, rel2id, schema)
    dev_feats = build_split(dev, char2id, upos2id, rel2id, schema)
    test_feats = build_split(test, char2id, upos2id, rel2id, schema)

    def make_batches(feats_list, bs, pad_id):
        return list(batchify_with_spans(feats_list, bs, pad_id))

    train_batches = make_batches(train_feats, args.batch_size, pad_id=char2id["<pad>"])
    dev_batches   = make_batches(dev_feats, args.batch_size, pad_id=char2id["<pad>"])
    test_batches  = make_batches(test_feats, args.batch_size, pad_id=char2id["<pad>"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []

    for seed in args.seeds:
        set_seed(seed)
        model = PosDepMorphCharModel(
            num_chars=len(char2id),
            num_upos=len(upos2id),
            num_rels=len(rel2id),
            schema=schema,
            emb_dim=args.emb_dim,
            enc_hidden=args.hidden,
            enc_layers=args.layers,
            enc_dropout=args.enc_dropout,
            mlp_dim=args.mlp_dim,
            mlp_dropout=args.mlp_dropout,
            morph_dropout=args.morph_dropout,
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
                "char2id": char2id,
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
            f"LaTeX row: CharBiLSTM + Biaffine & {agg['pos_acc_mean']:.2f}±{agg['pos_acc_std']:.2f} "
            f"& {agg['uas_mean']:.2f}±{agg['uas_std']:.2f} & {agg['las_mean']:.2f}±{agg['las_std']:.2f} "
            f"& {agg['morph_micro_acc_mean']:.2f}±{agg['morph_micro_acc_std']:.2f}\n"
        )

if __name__ == "__main__":
    main()
