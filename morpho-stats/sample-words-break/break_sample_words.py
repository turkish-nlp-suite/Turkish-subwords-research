#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from transformers import AutoTokenizer

# Fixed layout
COLLECTIONS = ["books", "minimal", "alldata"]
SIZES = ["2k", "5k", "10k", "20k", "32k", "52k", "128k"]
NAME_FMT = "turkish-nlp-suite/wordpiece_{}_cased_{}"

# Words to tokenize (edit as needed)
WORDS = [
    "ev",
    "evim",
    "evlerimiz",
    "evde",
    "evden",
    "evdeki",
    "evdekiler",
    "okudum",
    "okudu",
    "okumak",
    "okuma",
    "okudum",
    "okudular",
    "okuduklarımız",
    "okumadım",
    "evlerimiz",
    "evimiz",
    "evimizdeki",
    "evleriniz",
    "evlerimizden",
    "kitapçılardan",
    "gelmeyeceksiniz",
    "yazdırılmayacakmışsınız",
    "demokrasileştirmenin",
    "Ankara’dan",
    "İstanbul'uyla",
    "İstanbul'a",
    "İstanbul'da",
    "İstanbul'dan",
    "görülebilirdi",
    "çalıştırılabilir",
    "kapkara",
    "Ankara'ya",
    "Ankara'dan",
    "Ankara'da",
    "1923'te"
]

OUT_PATH = "broken_words.tsv"

def main():
    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        fp.write("\t".join(["collection", "size", "text", "tokens"]) + "\n")

        for collection in COLLECTIONS:
            for size in SIZES:
                tok_dir = NAME_FMT.format(size, collection)
                if not os.path.isdir(tok_dir):
                    print(f"[warn] missing: {tok_dir}")
                    continue

                try:
                    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True)
                except Exception:
                    tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=False)

                print(f"[info] {collection}/{size} started")
                for text in WORDS:
                    toks = tok.tokenize(text)  # simple, direct
                    # Write as a JSON-ish list for readability; no IDs, no offsets
                    fp.write("\t".join([
                        collection,
                        size,
                        text.replace("\t", " ").replace("\n", " "),
                        "[" + ", ".join(toks) + "]",
                    ]) + "\n")
                print(f"[info] {collection}/{size} ended")

if __name__ == "__main__":
    main()

