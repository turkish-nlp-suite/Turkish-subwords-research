#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from collections import Counter
from typing import Dict, List, Tuple, Set

def load_counts_ordered(path: str) -> List[str]:
    # Reads "token count" lines and returns tokens sorted by descending count.
    pairs = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                w, c_str = line.rsplit(" ", 1)
                c = int(c_str)
            except ValueError:
                continue
            if c > 0:
                pairs.append((w, c))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in pairs]

def load_counts_map(path: str) -> Counter:
    ctr = Counter()
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            try:
                w, c_str = line.rsplit(" ", 1)
                c = int(c_str)
            except ValueError:
                continue
            if c > 0:
                ctr[w] += c
    return ctr

def load_wcache(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def default_k_grid(maxk: int) -> List[int]:
    base = [50, 100, 200, 500, 1000, 2000, 5000]
    logs = []
    x = 3.0
    while True:
        val = int(round(10 ** x))
        if val > maxk:
            break
        logs.append(val)
        x += 0.3
    grid = sorted(set([k for k in base if k <= maxk] + logs + [maxk]))
    return grid

def resolve_paths(task: str, split: str) -> Dict[str, str]:
    paths = {
        "train_sub_counts": os.path.join(task, f"train_subwrd_counts.txt"),
        "wcache": os.path.join(task, f"wcache_{split}.json"),
        "eval_word_counts": os.path.join(task, f"{split}_word_counts.txt"),  # optional
        "eval_sub_counts": os.path.join(task, f"{split}_subwrd_counts.txt"),  # optional (for info)
        "out_dir": task,
    }
    if not os.path.exists(paths["train_sub_counts"]):
        raise FileNotFoundError(f"Missing train subword counts: {paths['train_sub_counts']}")
    if not os.path.exists(paths["wcache"]):
        raise FileNotFoundError(f"Missing wcache for split={split}: {paths['wcache']}")
    return paths

def compute_word_coverage_vs_k(
    train_subwords_ranked: List[str],
    wcache: Dict[str, List[str]],
    ks: List[int],
    eval_word_counts: Counter = None,
) -> List[Tuple[int, float, float]]:
    """
    Coverage is defined at word level:
      - A word is covered iff all its subwords are in top-K.
    Returns list of (K, token_coverage, type_coverage).
    token_coverage:
      - If eval_word_counts provided: sum_counts(covered_words)/sum_counts(all_words)
      - Else: equals type_coverage (so it's not misleading).
    """
    # Prepare eval vocab
    words = list(wcache.keys())
    word_subs: Dict[str, Set[str]] = {w: set(wcache[w]) for w in words}

    # Optional word counts
    total_tokens = None
    wc = None
    if eval_word_counts is not None and len(eval_word_counts) > 0:
        wc = eval_word_counts
        total_tokens = sum(wc.values())

    # Iterate K in ascending order, maintaining a growing set of allowed subwords
    maxk = len(train_subwords_ranked)
    ks_sorted = sorted(set(min(k, maxk) for k in ks))
    covered_state = {w: False for w in words}
    covered_types = 0
    covered_tokens = 0

    allowed = set()
    cursor = 0

    # For efficiency, precompute for each word the number of required subwords
    needed_counts = {w: len(s) for w, s in word_subs.items()}
    # Track how many of a word's subwords are currently included in allowed
    have_counts = {w: 0 for w in words}

    # Build reverse index: subword -> list of words that contain it
    sub_to_words: Dict[str, List[str]] = {}
    for w, subs in word_subs.items():
        for s in subs:
            sub_to_words.setdefault(s, []).append(w)

    results: List[Tuple[int, float, float]] = []

    for k in ks_sorted:
        while cursor < k:
            s = train_subwords_ranked[cursor]
            cursor += 1
            if s in allowed:
                continue
            allowed.add(s)
            for w in sub_to_words.get(s, []):
                if covered_state[w]:
                    continue
                have_counts[w] += 1
                if have_counts[w] == needed_counts[w]:
                    covered_state[w] = True
                    covered_types += 1
                    if total_tokens is not None:
                        covered_tokens += wc.get(w, 0)

        type_cov = covered_types / len(words) if words else 0.0
        if total_tokens is not None and total_tokens > 0:
            tok_cov = covered_tokens / total_tokens
        else:
            tok_cov = type_cov
        results.append((k, tok_cov, type_cov))

    return results

def find_k_for_targets(
    train_subwords_ranked: List[str],
    wcache: Dict[str, List[str]],
    targets: List[float],
    eval_word_counts: Counter = None,
) -> Dict[float, int]:
    """
    Minimal K achieving given token-coverage targets, where token-coverage is
    computed as word-level reconstructability (all subwords present).
    If eval_word_counts is None, token coverage = type coverage.
    """
    words = list(wcache.keys())
    word_subs: Dict[str, Set[str]] = {w: set(wcache[w]) for w in words}

    wc = eval_word_counts or Counter({w: 1 for w in words})
    total_tokens = sum(wc.values())
    if total_tokens == 0:
        return {t: 0 for t in targets}

    tgt_sorted = sorted(set(max(0.0, min(1.0, t)) for t in targets))
    need = [math.ceil(t * total_tokens) for t in tgt_sorted]
    res = {t: len(train_subwords_ranked) for t in tgt_sorted}

    # Incremental as K grows
    allowed = set()
    cursor = 0
    needed_counts = {w: len(s) for w, s in word_subs.items()}
    have_counts = {w: 0 for w in words}
    covered = {w: False for w in words}
    covered_tokens = 0
    ti = 0

    sub_to_words: Dict[str, List[str]] = {}
    for w, subs in word_subs.items():
        for s in subs:
            sub_to_words.setdefault(s, []).append(w)

    for k, s in enumerate(train_subwords_ranked, start=1):
        if s in allowed:
            continue
        allowed.add(s)
        for w in sub_to_words.get(s, []):
            if covered[w]:
                continue
            have_counts[w] += 1
            if have_counts[w] == needed_counts[w]:
                covered[w] = True
                covered_tokens += wc.get(w, 0)

        while ti < len(need) and covered_tokens >= need[ti]:
            res[tgt_sorted[ti]] = k
            ti += 1
        if ti >= len(need):
            break

    return res

def write_csv(out_csv: str, results: List[Tuple[int, float, float]]) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("k,token_coverage,type_coverage\n")
        for k, tok, typ in results:
            f.write(f"{k},{tok:.6f},{typ:.6f}\n")

def write_targets_csv(path: str, target_map: Dict[float, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("target_token_coverage,min_k\n")
        for t in sorted(target_map.keys()):
            f.write(f"{t:.6f},{target_map[t]}\n")

def pretty_print(results, target_map, task, split, n_words, total_tokens):
    print("\nSubword-based full-word coverage")
    print(f"Task: {task} | Split: {split}")
    print(f"Eval words (types): {n_words} | Token sum: {total_tokens if total_tokens is not None else 'NA'}")
    if not results:
        print("No results.")
        return
    w1 = max(3, max(len(str(k)) for k, _, _ in results))
    header = f"{'K'.rjust(w1)}  TokenCov  TypeCov"
    print(header)
    print("-" * len(header))
    for k, tok, typ in results:
        print(f"{str(k).rjust(w1)}  {tok:8.4%}  {typ:7.4%}")
    if target_map:
        print("\nTarget token-coverage -> minimal K")
        print("----------------------------------")
        for t in sorted(target_map.keys()):
            print(f"{t:.3f} -> K={target_map[t]}")
    compact = [f"K={k}: tok={tok*100:.2f}%, type={typ*100:.2f}%" for k, tok, typ in results]
    print("\nCompact:")
    print(" | ".join(compact))

def main():
    ap = argparse.ArgumentParser(
        description="Compute coverage vs K for words fully covered by top-K subwords."
    )
    ap.add_argument("--task", required=True, help="Task name (e.g., NER, cola, etc.)")
    ap.add_argument("--split", default="test", help="Eval split name")
    ap.add_argument("--k_list", type=str, default="", help="Comma-separated K list; default uses an auto grid.")
    ap.add_argument("--target_coverages", type=str, default="", help="Comma-separated targets (e.g., 0.8,0.9,0.95) over token coverage.")
    ap.add_argument("--out_csv", type=str, default="", help="Output CSV for coverage curve. Default: task/coverage_subw_{task}_{split}.csv")
    ap.add_argument("--word_counts_file", type=str, default="", help="Optional eval word counts file to weight token coverage.")
    args = ap.parse_args()

    paths = resolve_paths(args.task, args.split)

    train_sub_rank = load_counts_ordered(paths["train_sub_counts"])
    if not train_sub_rank:
        raise RuntimeError("Empty train subword counts ranking.")

    wcache = load_wcache(paths["wcache"])
    if not wcache:
        raise RuntimeError("Empty wcache; nothing to evaluate.")

    eval_word_counts = None
    total_tokens = None
    if args.word_counts_file and os.path.exists(args.word_counts_file):
        eval_word_counts = load_counts_map(args.word_counts_file)
        total_tokens = sum(eval_word_counts.values())

    ks = [int(x) for x in args.k_list.split(",")] if args.k_list.strip() else default_k_grid(len(train_sub_rank))

    # Targets
    target_map = {}
    if args.target_coverages.strip():
        targets = [float(x) for x in args.target_coverages.split(",") if x.strip()]
        target_map = find_k_for_targets(train_sub_rank, wcache, targets, eval_word_counts)
        ks = sorted(set(ks + list(target_map.values())))

    results = compute_word_coverage_vs_k(train_sub_rank, wcache, ks, eval_word_counts)

    out_csv = args.out_csv or os.path.join(paths["out_dir"], f"coverage_subw_{args.split}.csv")
    write_csv(out_csv, results)
    print(f"\nWrote CSV: {out_csv}")

    if target_map:
        tgt_path = out_csv.replace(".csv", "_targets.csv")
        write_targets_csv(tgt_path, target_map)
        print(f"Wrote target mapping: {tgt_path}")

    n_words = len(wcache)
    pretty_print(results, target_map, args.task, args.split, n_words, total_tokens)

if __name__ == "__main__":
    main()
