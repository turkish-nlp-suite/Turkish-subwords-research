#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import argparse
from pathlib import Path

def smart_cov_name(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s if s else "0"

def read_tokens_from_counts(counts_file: Path):
    # Reads tokens in train rank order (first column per line)
    toks = []
    with counts_file.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks.append(line.rsplit(" ", 1)[0])
    return toks

def main():
    ap = argparse.ArgumentParser(
        description="Write top-K subword vocab files using target-K CSVs."
    )
    ap.add_argument("--task", required=True)
    ap.add_argument("--out_dir", required=True, help="Directory to write vocab files into")
    ap.add_argument("--coverage_targets_csv", default=None, help="coverage_subw_<task>_<split>_targets.csv")
    ap.add_argument("--train_sub_counts", default=None, help="train_subwrd_counts_<task>.txt")
    ap.add_argument("--prefix", default="top_", help="Output filename prefix")
    args = ap.parse_args()

    cov_csv = Path(args.coverage_targets_csv) if args.coverage_targets_csv else Path(f"{args.task}/coverage_subw_{args.task}_train_targets.csv")
    counts_file = Path(args.train_sub_counts) if args.train_sub_counts else Path(f"{args.task}/train_subwrd_counts_{args.task}.txt")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cov_csv.exists():
        raise FileNotFoundError(f"Missing {cov_csv}")
    if not counts_file.exists():
        raise FileNotFoundError(f"Missing {counts_file}")

    # Read train subword tokens in rank order
    tokens = read_tokens_from_counts(counts_file)

    # Read coverage targets
    rows = []
    with cov_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "target_token_coverage" in reader.fieldnames:
            for r in reader:
                rows.append((float(r["target_token_coverage"]), int(r["min_k"])))
        else:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cov_str, k_str = line.split(",")[:2]
                rows.append((float(cov_str), int(k_str)))

    seen_k = set()
    written = 0
    for cov, k in sorted(rows, key=lambda x: x[0]):
        if k in seen_k:
            continue
        seen_k.add(k)
        cov_name = smart_cov_name(cov)
        out_path = out_dir / f"{args.prefix}{cov_name}.txt"
        with out_path.open("w", encoding="utf-8") as out:
            for tok in tokens[:k]:
                out.write(tok + "\n")
        print(f"Wrote {out_path} (K={k}, coverage={cov_name})")
        written += 1

    print(f"Done. Wrote {written} vocab files to {out_dir}")

if __name__ == "__main__":
    main()
