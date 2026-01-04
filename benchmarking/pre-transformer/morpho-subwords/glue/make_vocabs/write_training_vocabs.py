#!/usr/bin/env python3
import csv
import argparse
from pathlib import Path

def smart_cov_name(x: float) -> str:
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cola")
    ap.add_argument("--out_dir", default="../training/vocab_files")
    ap.add_argument("--coverage_csv", default=None, help="coverage_<task>_validation_targets.csv")
    ap.add_argument("--counts_file", default=None, help="train_word_counts_<task>.txt")
    args = ap.parse_args()

    cov_csv = Path(args.coverage_csv) if args.coverage_csv else Path(f"{args.task}/coverage_{args.task}_train_targets.csv")
    counts_file = Path(args.counts_file) if args.counts_file else Path(f"{args.task}/train_word_counts_{args.task}.txt")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cov_csv.exists():
        raise FileNotFoundError(f"Missing {cov_csv}")
    if not counts_file.exists():
        raise FileNotFoundError(f"Missing {counts_file}")

    # Read first-column tokens from counts file into a list (lazy streaming)
    # We’ll head N lines each time without re-reading fully by seeking from start.
    # Simpler: read once all tokens (safe if file is small; otherwise stream per K).
    tokens = []
    with counts_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            tok = line.split()[0]
            tokens.append(tok)

    # Load coverage targets
    # Expect header: target_token_coverage,min_k
    rows = []
    with cov_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # If no header, fall back to manual parse
        if reader.fieldnames is None or "target_token_coverage" not in reader.fieldnames:
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cov_str, k_str = line.split(",")[:2]
                rows.append((float(cov_str), int(k_str)))
        else:
            for r in reader:
                rows.append((float(r["target_token_coverage"]), int(r["min_k"])))

    # Deduplicate by min_k (keep first occurrence)
    seen_k = set()
    dedup = []
    for cov, k in rows:
        if k in seen_k:
            continue
        seen_k.add(k)
        dedup.append((cov, k))

    # Create files
    for cov, k in dedup:
        cov_name = smart_cov_name(cov)
        out_path = out_dir / f"top_{cov_name}.txt"
        print(f"Writing {out_path} (first {k} tokens)")
        with out_path.open("w", encoding="utf-8") as out:
            for tok in tokens[:k]:
                out.write(tok + "\n")

    print(f"Done. Wrote {len(dedup)} vocab files to {out_dir}")

if __name__ == "__main__":
    main()
