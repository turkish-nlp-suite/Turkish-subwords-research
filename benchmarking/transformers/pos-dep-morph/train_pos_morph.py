#!/usr/bin/env python3
import argparse, os, json, random
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoConfig, AutoModel,
                          Trainer, TrainingArguments, set_seed)

# -----------------------
# Data loading: CoNLL-U (now includes FEATS)
# -----------------------
def read_conllu(path: str):
    sents = []
    tokens, upos, heads, rels, feats = [], [], [], [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if tokens:
                    sents.append({"tokens": tokens, "upos": upos, "heads": heads, "rels": rels, "feats": feats})
                    tokens, upos, heads, rels, feats = [], [], [], [], []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:
                # Skip multiword/empty nodes.
                continue
            tok = cols[1]
            up = cols[3]               # UPOS
            feat = cols[5]             # FEATS raw ("_" or "A=B|C=D")
            head = int(cols[6])        # HEAD index (0=root)
            rel = cols[7]              # DEPREL
            tokens.append(tok)
            upos.append(up)
            heads.append(head)
            rels.append(rel)
            feats.append(feat)
    if tokens:
        sents.append({"tokens": tokens, "upos": upos, "heads": heads, "rels": rels, "feats": feats})
    return sents

# -----------------------
# Morph helpers (embedded from your preprocess_morph.py)
# -----------------------
def parse_feats_str(feats_str: str) -> Dict[str, str]:
    if feats_str == "_" or feats_str.strip() == "":
        return {}
    kv = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            kv[k] = v
    return kv

class MorphSchema:
    def __init__(self, attrs, values, value2id, id2value):
        self.attrs = attrs
        self.values = values
        self.value2id = value2id
        self.id2value = id2value

def build_schema(train_feats: List[List[str]]):
    attrs = set()
    values = {}
    for sent in train_feats:
        for f in sent:
            feats = parse_feats_str(f)
            for k, v in feats.items():
                attrs.add(k)
                values.setdefault(k, set()).add(v)
    attrs = sorted(list(attrs))
    for a in attrs:
        values.setdefault(a, set())
        values[a].add("None")
        values[a] = sorted(values[a])
    value2id = {a: {v: i for i, v in enumerate(values[a])} for a in attrs}
    id2value = {a: {i: v for i, v in enumerate(values[a])} for a in attrs}
    return MorphSchema(attrs=attrs, values=values, value2id=value2id, id2value=id2value)

def to_attr_ids(feats_str: str, schema: MorphSchema):
    feats = parse_feats_str(feats_str)
    y = {}
    for a in schema.attrs:
        val = feats.get(a, "None")
        y[a] = schema.value2id[a].get(val, schema.value2id[a]["None"])
    return y

# -----------------------
# Dataset + preprocessing (adds morphology labels)
# -----------------------
class PosDepDataset(Dataset):
    def __init__(self, examples, tokenizer, upos2id, rel2id, schema: Optional[MorphSchema] = None, max_length=256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.upos2id = upos2id
        self.rel2id = rel2id
        self.schema = schema
        self.max_length = max_length
        self.items = [self._encode(ex) for ex in examples]

    def _encode(self, ex):
        enc = self.tokenizer(
            ex["tokens"], is_split_into_words=True,
            truncation=True, max_length=self.max_length,
            return_attention_mask=True
        )
        word_ids = enc.word_ids()

        # POS labels aligned to first subword; -100 elsewhere
        pos_labels = []
        prev_w = None
        for wi in word_ids:
            if wi is None:
                pos_labels.append(-100)
            elif wi != prev_w:
                pos_labels.append(self.upos2id[ex["upos"][wi]])
            else:
                pos_labels.append(-100)
            prev_w = wi

        # Map word -> first subword token index
        word_to_tok = {}
        for i, wi in enumerate(word_ids):
            if wi is None:
                continue
            if wi not in word_to_tok:
                word_to_tok[wi] = i

        input_ids = enc["input_ids"]
        cls_index = 0
        num_tokens = len(input_ids)

        n_words = len(ex["tokens"])
        active_word_idx = sorted([w for w in range(n_words) if w in word_to_tok])

        arc_targets = np.full(num_tokens, -100, dtype=np.int64)
        rel_targets = np.full(num_tokens, -100, dtype=np.int64)

        for w in active_word_idx:
            tok_idx = word_to_tok[w]
            head_w = ex["heads"][w]  # 0..n
            if head_w == 0:
                head_tok = cls_index
            else:
                hw = head_w - 1  # UD heads are 1-based for words
                head_tok = word_to_tok.get(hw, None)
            if head_tok is not None:
                arc_targets[tok_idx] = head_tok
                rel_targets[tok_idx] = self.rel2id[ex["rels"][w]]

        # Morphology labels per attribute, aligned to first subword
        if self.schema is not None:
            morph_labels = {}
            # Build word-level attr ids
            y_words_by_attr = {a: [] for a in self.schema.attrs}
            for feats_str in ex["feats"]:
                y_attr = to_attr_ids(feats_str, self.schema)
                for a in self.schema.attrs:
                    y_words_by_attr[a].append(y_attr[a])
            # Align to token space
            for a in self.schema.attrs:
                seq = []
                prev_w = None
                for wi in word_ids:
                    if wi is None:
                        seq.append(-100)
                    elif wi != prev_w:
                        seq.append(y_words_by_attr[a][wi])
                    else:
                        seq.append(-100)
                    prev_w = wi
                morph_labels[a] = seq
        else:
            morph_labels = None

        out = {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels_pos": pos_labels,
            "labels_arc": arc_targets.tolist(),
            "labels_rel": rel_targets.tolist(),
            "word_ids": [wi if wi is not None else -1 for wi in word_ids],
        }
        if morph_labels is not None:
            for a in self.schema.attrs:
                out[f"labels_morph__{a}"] = morph_labels[a]
        return out

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# -----------------------
# Model: encoder + biaffine + morphology head
# -----------------------
class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.U = nn.Parameter(torch.zeros(in_dim + (1 if bias_x else 0), out_dim, in_dim + (1 if bias_y else 0)))
        nn.init.xavier_uniform_(self.U)

    def forward(self, x, y):
        # x: [B,T,Dx], y: [B,T,Dy]
        if self.bias_x:
            ones = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, ones], dim=-1)
        if self.bias_y:
            ones = torch.ones(y.size(0), y.size(1), 1, device=y.device, dtype=y.dtype)
            y = torch.cat([y, ones], dim=-1)
        # [B,T,Dx+] @ [Dx+,C,Dy+] -> [B,T,C,Dy+]
        xU = torch.einsum("bti, icj -> btcj", x, self.U)
        # [B,T,C,Dy+] x [B,T,Dy+]^T over last dim: -> [B,T,C,T]
        scores = torch.einsum("btcj, bkj -> btck", xU, y)
        return scores

class MultiAttrMorphHead(nn.Module):
    def __init__(self, hidden_size: int, schema: MorphSchema, dropout=0.1):
        super().__init__()
        self.schema = schema
        self.dropout = nn.Dropout(dropout)
        self.classifiers = nn.ModuleDict({
            a: nn.Linear(hidden_size, len(schema.values[a])) for a in schema.attrs
        })

    def forward(self, hidden_states, labels: Optional[Dict[str, torch.Tensor]] = None, attention_mask=None):
        x = self.dropout(hidden_states)  # [B, T, H]
        logits = {a: self.classifiers[a](x) for a in self.schema.attrs}  # each [B, T, Va]
        loss = None
        if labels is not None and len(labels) > 0:
            losses = []
            for a in self.schema.attrs:
                if a not in labels:
                    continue
                l = nn.functional.cross_entropy(
                    logits[a].view(-1, logits[a].size(-1)),
                    labels[a].view(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
                losses.append(l)
            if len(losses) > 0:
                loss = torch.stack(losses).mean()
        return logits, loss

class PosDepMorphModel(nn.Module):
    def __init__(self, model_name, num_upos, num_rels, schema: Optional[MorphSchema],
                 hidden_mlp=400, dropout=0.33, morph_dropout=0.1, morph_loss_weight=1.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hid = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # POS head
        self.pos_classifier = nn.Linear(hid, num_upos)
        # Arc/Rel MLPs
        self.mlp_arc_dep = nn.Sequential(nn.Linear(hid, hidden_mlp), nn.ReLU(), nn.Dropout(dropout))
        self.mlp_arc_head = nn.Sequential(nn.Linear(hid, hidden_mlp), nn.ReLU(), nn.Dropout(dropout))
        self.mlp_rel_dep = nn.Sequential(nn.Linear(hid, hidden_mlp), nn.ReLU(), nn.Dropout(dropout))
        self.mlp_rel_head = nn.Sequential(nn.Linear(hid, hidden_mlp), nn.ReLU(), nn.Dropout(dropout))
        # Biaffines
        self.arc_biaffine = Biaffine(hidden_mlp, 1)
        self.rel_biaffine = Biaffine(hidden_mlp, num_rels)
        # Morphology
        self.schema = schema
        self.morph_loss_weight = morph_loss_weight
        if schema is not None:
            self.morph_head = MultiAttrMorphHead(hid, schema, dropout=morph_dropout)
        else:
            self.morph_head = None

    def forward(self, input_ids=None, attention_mask=None,
                labels_pos=None, labels_arc=None, labels_rel=None, word_ids=None, **kwargs):
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(enc.last_hidden_state)  # [B,T,H]

        # POS logits
        pos_logits = self.pos_classifier(h)  # [B,T,num_upos]

        # Arc/Rel scores
        dep_arc = self.mlp_arc_dep(h)
        head_arc = self.mlp_arc_head(h)
        dep_rel = self.mlp_rel_dep(h)
        head_rel = self.mlp_rel_head(h)
        arc_scores = self.arc_biaffine(dep_arc, head_arc).squeeze(2)  # [B,T,T]
        rel_scores = self.rel_biaffine(dep_rel, head_rel)             # [B,T,num_rels,T]

        outputs = {"pos_logits": pos_logits, "arc_scores": arc_scores, "rel_scores": rel_scores}

        total_loss = None
        if labels_pos is not None and labels_arc is not None and labels_rel is not None:
            pos_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                pos_logits.view(-1, pos_logits.size(-1)), labels_pos.view(-1)
            )
            # Arc loss only at first-subword positions
            mask = (labels_arc != -100) & (attention_mask == 1)
            if mask.any():
                arc_loss = nn.CrossEntropyLoss()(arc_scores[mask], labels_arc[mask])
                # relation loss gathered at gold heads
                idxs = mask.nonzero(as_tuple=False)  # [N,2]
                b_idx = idxs[:, 0]
                t_idx = idxs[:, 1]
                gold_heads = labels_arc[b_idx, t_idx]  # [N]
                rel_logits = rel_scores[b_idx, t_idx, :, gold_heads]  # [N,num_rels]
                rel_loss = nn.CrossEntropyLoss()(rel_logits, labels_rel[b_idx, t_idx])
            else:
                arc_loss = torch.tensor(0.0, device=h.device)
                rel_loss = torch.tensor(0.0, device=h.device)
            total_loss = pos_loss + arc_loss + rel_loss
            outputs.update({"pos_loss": pos_loss, "arc_loss": arc_loss, "rel_loss": rel_loss})

        # Morph
        if self.morph_head is not None:
            labels_dict = {}
            have_any = False
            for a in self.schema.attrs:
                key = f"labels_morph__{a}"
                if key in kwargs and kwargs[key] is not None:
                    labels_dict[a] = kwargs[key]
                    have_any = True
            morph_logits, m_loss = self.morph_head(h, labels=labels_dict if have_any else None,
                                                   attention_mask=attention_mask)
            outputs["morph_logits"] = morph_logits
            if m_loss is not None:
                if total_loss is None:
                    total_loss = self.morph_loss_weight * m_loss
                else:
                    total_loss = total_loss + self.morph_loss_weight * m_loss
                outputs["morph_loss"] = m_loss

        if total_loss is not None:
            outputs["loss"] = total_loss
        return outputs

# -----------------------
# Collator with dynamic morph keys
# -----------------------
def default_data_collator(tokenizer, schema: Optional[MorphSchema] = None):
    def collate(features):
        batch = {}
        pad = tokenizer.pad(
            {k: [f[k] for f in features] for k in ["input_ids", "attention_mask"]},
            return_tensors="pt"
        )
        batch.update(pad)

        def pad_sequence(key, pad_value=-100):
            seqs = [torch.tensor(f[key], dtype=torch.long) for f in features]
            maxlen = batch["input_ids"].size(1)
            out = torch.full((len(seqs), maxlen), pad_value, dtype=torch.long)
            for i, s in enumerate(seqs):
                L = min(maxlen, s.size(0))
                out[i, :L] = s[:L]
            return out

        batch["labels_pos"] = pad_sequence("labels_pos", -100)
        batch["labels_arc"] = pad_sequence("labels_arc", -100)
        batch["labels_rel"] = pad_sequence("labels_rel", -100)
        batch["word_ids"] = pad_sequence("word_ids", -1)

        if schema is not None:
            for a in schema.attrs:
                key = f"labels_morph__{a}"
                if key in features[0]:  # present in dataset
                    batch[key] = pad_sequence(key, -100)

        return batch
    return collate

# -----------------------
# Evaluation: POS acc, UAS, LAS, Morph per-attr + micro acc
# -----------------------
@torch.no_grad()
def evaluate(model, dataset, tokenizer, id2upos, id2rel, device, schema: Optional[MorphSchema] = None):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False,
        collate_fn=default_data_collator(tokenizer, schema)
    )
    total_pos, correct_pos = 0, 0
    total_arcs, correct_uas, correct_las = 0, 0, 0

    morph_correct = {a: 0 for a in (schema.attrs if schema is not None else [])}
    morph_total = {a: 0 for a in (schema.attrs if schema is not None else [])}

    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(device)
        out = model(**batch)
        pos_pred = out["pos_logits"].argmax(-1)  # [B,T]
        arc_scores = out["arc_scores"]           # [B,T,T]
        rel_scores = out["rel_scores"]           # [B,T,num_rels,T]

        # POS accuracy
        mask_pos = batch["labels_pos"] != -100
        correct_pos += (pos_pred[mask_pos] == batch["labels_pos"][mask_pos]).sum().item()
        total_pos += mask_pos.sum().item()

        # UAS/LAS at first-subword positions
        mask_arc = batch["labels_arc"] != -100
        if mask_arc.any():
            pred_heads = arc_scores.argmax(-1)  # [B,T]
            idxs = mask_arc.nonzero(as_tuple=False)
            for b, t in idxs:
                gold_h = int(batch["labels_arc"][b, t].item())
                pred_h = int(pred_heads[b, t].item())
                total_arcs += 1
                if pred_h == gold_h:
                    correct_uas += 1
                    rel_logits = rel_scores[b, t, :, pred_h]
                    pred_rel = int(rel_logits.argmax(-1).item())
                    gold_rel = int(batch["labels_rel"][b, t].item())
                    if pred_rel == gold_rel:
                        correct_las += 1

        # Morphology accuracy
        if schema is not None and "morph_logits" in out:
            for a in schema.attrs:
                key = f"labels_morph__{a}"
                if key not in batch:
                    continue
                logits = out["morph_logits"][a]  # [B,T,Va]
                preds = logits.argmax(-1)
                mask = batch[key] != -100
                morph_correct[a] += (preds[mask] == batch[key][mask]).sum().item()
                morph_total[a] += mask.sum().item()

    pos_acc = correct_pos / max(1, total_pos)
    uas = correct_uas / max(1, total_arcs)
    las = correct_las / max(1, total_arcs)

    metrics = {"pos_acc": pos_acc, "uas": uas, "las": las}
    if schema is not None and len(schema.attrs) > 0:
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.33)
    ap.add_argument("--hidden_mlp", type=int, default=400)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--output_dir", default="posdep_out")
    ap.add_argument("--morph_loss_weight", type=float, default=1.0)
    args = ap.parse_args()

    train = read_conllu(args.train)
    dev = read_conllu(args.dev)
    test = read_conllu(args.test)

    # Label spaces
    upos_set = sorted({t for ex in train + dev + test for t in ex["upos"]})
    rel_set = sorted({r for ex in train + dev + test for r in ex["rels"]})
    upos2id = {u: i for i, u in enumerate(upos_set)}
    id2upos = {i: u for u, i in upos2id.items()}
    rel2id = {r: i for i, r in enumerate(rel_set)}
    id2rel = {i: r for r, i in rel2id.items()}

    # Morphology schema from train FEATS
    train_feats_seqs = [ex["feats"] for ex in train]
    schema = build_schema(train_feats_seqs)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_ds = PosDepDataset(train, tokenizer, upos2id, rel2id, schema=schema, max_length=args.max_length)
    dev_ds = PosDepDataset(dev, tokenizer, upos2id, rel2id, schema=schema, max_length=args.max_length)
    test_ds = PosDepDataset(test, tokenizer, upos2id, rel2id, schema=schema, max_length=args.max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    for seed in args.seeds:
        set_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = PosDepMorphModel(
            args.model, num_upos=len(upos2id), num_rels=len(rel2id),
            schema=schema, hidden_mlp=args.hidden_mlp, dropout=args.dropout,
            morph_dropout=0.1, morph_loss_weight=args.morph_loss_weight
        ).to(device)

        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"seed{seed}"),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=max(4, args.batch_size),
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_strategy="steps",
            logging_steps=100,
            load_best_model_at_end=False,
            fp16=args.fp16,
            bf16=args.bf16,
            report_to=["none"],
        )

        collate_fn = default_data_collator(tokenizer, schema=schema)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=tokenizer,
            data_collator=collate_fn,
        )

        trainer.train()

        dev_metrics = evaluate(model, dev_ds, tokenizer, id2upos, id2rel, device, schema=schema)
        test_metrics = evaluate(model, test_ds, tokenizer, id2upos, id2rel, device, schema=schema)

        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(os.path.join(training_args.output_dir, "dev_metrics.json"), "w") as f:
            json.dump(dev_metrics, f, indent=2)
        with open(os.path.join(training_args.output_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)

        print(f"Seed {seed} dev:", dev_metrics)
        print(f"Seed {seed} test:", test_metrics)
        results.append(test_metrics)

    # Aggregate across seeds
    pos_accs = [r["pos_acc"] for r in results]
    uass = [r["uas"] for r in results]
    lass = [r["las"] for r in results]
    morph_micro = [r.get("morph_micro_acc", 0.0) for r in results]

    agg = {
        "pos_acc_mean": float(np.mean(pos_accs)),
        "pos_acc_std": float(np.std(pos_accs)),
        "uas_mean": float(np.mean(uass)),
        "uas_std": float(np.std(uass)),
        "las_mean": float(np.mean(lass)),
        "las_std": float(np.std(lass)),
        "morph_micro_acc_mean": float(np.mean(morph_micro)) if len(morph_micro) > 0 else 0.0,
        "morph_micro_acc_std": float(np.std(morph_micro)) if len(morph_micro) > 0 else 0.0,
        "seeds": args.seeds,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "final.txt")
    with open(final_path, "w") as ofile:
        ofile.write(json.dumps(agg, indent=2) + "\n")
        ofile.write(
            f"LaTeX row: {args.model} & {agg['pos_acc_mean']:.2f}±{agg['pos_acc_std']:.2f} "
            f"& {agg['uas_mean']:.2f}±{agg['uas_std']:.2f} & {agg['las_mean']:.2f}±{agg['las_std']:.2f} "
            f"& {agg['morph_micro_acc_mean']:.2f}±{agg['morph_micro_acc_std']:.2f}\n"
        )

if __name__ == "__main__":
    main()
