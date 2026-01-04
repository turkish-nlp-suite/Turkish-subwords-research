#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import re
from collections import Counter
from typing import List, Tuple, Set, Dict


def load_counts(path: str) -> Tuple[List[str], List[int]]:
    """
    Load a counts file with lines: 'token count'
    Returns:
      - tokens sorted by descending count (train ranking)
      - corresponding counts list
    """
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
                # Skip malformed lines
                continue
            pairs.append((w, c))
    pairs.sort(key=lambda x: x[1], reverse=True)
    words = [w for w, _ in pairs]
    counts = [c for _, c in pairs]
    return words, counts


def load_counts_map(path: str, lowercase: bool = False, keep_punct: bool = True) -> Counter:
    """
    Load a counts file ('token count') into a Counter.
    Applies optional lowercase and punctuation filtering to the token string.
    Counts are aggregated after normalization.
    """
    tok_re = re.compile(r"\w+", flags=re.UNICODE)
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
            t = w
            if lowercase:
                t = t.lower()
            if not keep_punct:
                m = tok_re.fullmatch(t)
                if not m:
                    continue
                t = m.group(0)
            if not t:
                continue
            if c > 0:
                ctr[t] += c
    return ctr


def coverage_curve_from_counts(train_words: List[str],
                               test_counts: Counter,
                               ks: List[int]) -> List[Tuple[int, float, float]]:
    """
    Compute coverage using top-K train_words and test_counts (Counter: token->count).
    - Token coverage: sum_{w in topK} test_counts[w] / total_test_tokens
    - Type coverage: |{w in topK and in test types}| / |test types|
    """
    test_types: Set[str] = set(test_counts.keys())
    total_tokens = sum(test_counts.values())
    total_types = len(test_types)
    if total_tokens == 0 or total_types == 0:
        return [(k, 0.0, 0.0) for k in sorted(set(ks))]

    results: List[Tuple[int, float, float]] = []
    top_set: Set[str] = set()
    covered_tokens = 0
    covered_types = 0
    seen_type = set()

    maxk = len(train_words)
    ks_sorted = sorted(set(min(k, maxk) for k in ks))

    # We incrementally grow top_set and maintain coverage accumulators
    cursor = 0
    for k in ks_sorted:
        while cursor < k:
            w = train_words[cursor]
            cursor += 1
            if w in test_counts:
                covered_tokens += test_counts[w]
                if w not in seen_type:
                    covered_types += 1
                    seen_type.add(w)
            top_set.add(w)
        token_cov = covered_tokens / total_tokens
        type_cov = covered_types / total_types
        results.append((k, token_cov, type_cov))
    return results


def default_k_grid(maxk: int) -> List[int]:
    """
    Reasonable default grid: small linear + log-spaced + include maxk.
    """
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


def find_k_for_targets_from_counts(train_words: List[str],
                                   test_counts: Counter,
                                   targets: List[float]) -> Dict[float, int]:
    """
    Minimal K to reach each target token-coverage using test_counts.
    Uses a single pass aligned to train ranking.
    """
    total_tokens = sum(test_counts.values())
    if total_tokens == 0:
        return {t: 0 for t in targets}

    tgt_sorted = sorted(set(max(0.0, min(1.0, t)) for t in targets))
    need = [int(math.ceil(t * total_tokens)) for t in tgt_sorted]
    results: Dict[float, int] = {t: len(train_words) for t in tgt_sorted}

    cum = 0
    ti = 0
    for k, w in enumerate(train_words, start=1):
        inc = test_counts.get(w, 0)
        if inc:
            cum += inc
        while ti < len(need) and cum >= need[ti]:
            results[tgt_sorted[ti]] = k
            ti += 1
        if ti >= len(need):
            break
    return results


def write_csv(out_csv: str, results: List[Tuple[int, float, float]]) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("k,token_coverage,type_coverage\n")
        for k, tok, typ in results:
            f.write(f"{k},{tok:.6f},{typ:.6f}\n")


def write_targets_csv(tgt_path: str, target_map: Dict[float, int]) -> None:
    os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
    with open(tgt_path, "w", encoding="utf-8") as f:
        f.write("target_token_coverage,min_k\n")
        for t in sorted(target_map.keys()):
            f.write(f"{t:.6f},{target_map[t]}\n")


def resolve_paths_counts_only(task: str, base_dir: str, split: str) -> Dict[str, str]:
    """
    Counts-only layout:
      {base_dir}/word/glue/{task}/train_word_counts_{task}.txt
      {base_dir}/word/glue/{task}/{split}_word_counts_{task}.txt
    """
    paths = {
        "train_counts": os.path.join(task, f"train_word_counts_{task}.txt"),
        "test_counts": os.path.join(task, f"{split}_word_counts_{task}.txt"),
        "out_dir": task,
    }
    if not os.path.exists(paths["train_counts"]):
        raise FileNotFoundError(f"Train counts file not found: {paths['train_counts']}")
    if not os.path.exists(paths["test_counts"]):
        raise FileNotFoundError(f"Eval counts file not found: {paths['test_counts']}")
    return paths


def pretty_print(results: List[Tuple[int, float, float]],
                 target_map: Dict[float, int],
                 task: str, split: str,
                 total_tokens: int, total_types: int) -> None:
    print("\nCoverage results")
    print(f"Task: {task} | Split: {split}")
    print(f"Tokens: {total_tokens} | Unique types: {total_types}")
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
        description="Coverage vs. top-K vocabulary using counts-only files; pretty prints results."
    )
    ap.add_argument("--task", required=True, help="Task name (e.g., cola, sst2, mnli, mrpc, stsb, ner, boun)")
    ap.add_argument("--base_dir", required=True, help="Base directory containing word/glue/{task}/ files")
    ap.add_argument("--split", default="test", help="Which split to evaluate")
    ap.add_argument("--lowercase", default=False, action="store_true", help="Lowercase tokens before matching")
    ap.add_argument("--keep_punct", default=False, action="store_true", help="Keep punctuation tokens (default: False)")
    ap.add_argument("--k_list", type=str, default="", help="Comma-separated K list (e.g., 100,500,1000). If empty, auto grid.")
    ap.add_argument("--target_coverages", type=str, default="", help="Comma-separated token-coverage targets (e.g., 0.8,0.9,0.95)")
    ap.add_argument("--out_csv", type=str, default="", help="Output CSV path. Default: auto in task dir.")
    args = ap.parse_args()

    paths = resolve_paths_counts_only(args.task, args.base_dir, args.split)

    # Train ranking
    train_words, _ = load_counts(paths["train_counts"])

    # Eval counts (with normalization options)
    test_counts = load_counts_map(paths["test_counts"], lowercase=args.lowercase, keep_punct=args.keep_punct)
    total_tokens = sum(test_counts.values())
    total_types = len(test_counts)
    if total_tokens == 0 or total_types == 0:
        raise RuntimeError(f"No counts read from {paths['test_counts']} after normalization.")

    # K grid
    if args.k_list:
        ks = [int(x) for x in args.k_list.split(",") if x.strip()]
    else:
        ks = default_k_grid(len(train_words))

    # Targets (minimal K for desired token coverage)
    target_map: Dict[float, int] = {}
    if args.target_coverages.strip():
        targets = [float(x) for x in args.target_coverages.split(",") if x.strip()]
        target_map = find_k_for_targets_from_counts(train_words, test_counts, targets)
        ks = sorted(set(ks + list(target_map.values())))

    # Coverage curve
    results = coverage_curve_from_counts(train_words, test_counts, ks)

    # Pretty print
    pretty_print(results, target_map, args.task, args.split, total_tokens, total_types)

    # Write CSVs
    out_csv = args.out_csv or os.path.join(paths["out_dir"], f"coverage_{args.task}_{args.split}.csv")
    write_csv(out_csv, results)
    print(f"\nWrote CSV: {out_csv}")

    if target_map:
        tgt_path = out_csv.replace(".csv", "_targets.csv")
        write_targets_csv(tgt_path, target_map)
        print(f"Wrote target mapping: {tgt_path}")


if __name__ == "__main__":
    main()
