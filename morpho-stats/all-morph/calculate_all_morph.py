import os
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any

from datasets import load_dataset
from transformers import AutoTokenizer
from jiwer import wer, mer, wil, wip, cer

# -----------------------
# Data loading (your splits)
# -----------------------
dset_cekimli = load_dataset("../Dataset", "cekimli", split="validation")
dset_nouns = load_dataset("../Dataset", "common_noun", split="validation")
dset_verbs = load_dataset("../Dataset", "common_verb", split="validation")

dset_lemmas = load_dataset("../Dataset", "lemma", split="validation")
dset_common_lemmas = load_dataset("../Dataset", "common_lemma", split="validation")

# -----------------------
# Utilities
# -----------------------

def wp_strip_prefix(pieces: List[str]) -> List[str]:
    # For WordPiece tokenizers that use '##' continuation markers.
    return [p.lstrip("##") for p in pieces]

def apply_consonant_alteration(word: str, lemma: str) -> str:
    # If lemma not prefix of word, approximate unsuz yumuşaması by copying last char from word prefix.
    if not word.startswith(lemma) and len(lemma) > 0:
        llen = len(lemma)
        root = word[:llen]
        if len(root) > 0:
            lemma = lemma[:-1] + root[-1]
    return lemma

def gold_morpheme_segments(lemma: str, suffixes: str, word: str) -> Tuple[List[str], List[int]]:
    """
    Returns gold morpheme list and gold boundary indices (character positions in word)
    Boundaries include end of each morpheme starting from 0 at string start.
    """
    suffix_list = suffixes.split("+") if suffixes else []
    # Allow consonant alternation fix for lemma boundary detection
    adj_lemma = lemma if word.startswith(lemma) else apply_consonant_alteration(word, lemma)

    morphemes = [adj_lemma] + suffix_list
    # Compute boundaries cumulatively by length of morpheme strings
    # We assume simple concatenation equals the surface (after the lemma tweak).
    boundaries = []
    pos = 0
    for m in morphemes:
        pos += len(m)
        boundaries.append(pos)
    return morphemes, boundaries  # boundaries: positions after each morpheme

def predicted_boundaries(tokenizer, word: str) -> Tuple[List[str], List[int]]:
    pieces = tokenizer.tokenize(word)
    core = wp_strip_prefix(pieces)
    # Build character-length spans by concatenation
    boundaries = []
    pos = 0
    for p in core:
        pos += len(p)
        boundaries.append(pos)
    return core, boundaries

def boundary_prf(gold: List[int], pred: List[int]) -> Tuple[float, float, float]:
    gold_set = set(gold)
    pred_set = set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1

def lemma_single_token(lemma: str, tokenizer) -> int:
    parts = wp_strip_prefix(tokenizer.tokenize(lemma))
    return 1 if len(parts) == 1 else 0

def lemma_boundary_hit(word: str, lemma: str, tokenizer) -> int:
    # Is there a predicted boundary exactly at len(adjusted_lemma)?
    adj_lemma = lemma if word.startswith(lemma) else apply_consonant_alteration(word, lemma)
    _, pred_b = predicted_boundaries(tokenizer, word)
    return 1 if (len(adj_lemma) in set(pred_b)) else 0

def jiwer_seq_metrics(analyses: List[str], parts: List[str]) -> Dict[str, float]:
    return {
        "WER": wer(analyses, parts),
        "MER": mer(analyses, parts),
        "WIL": wil(analyses, parts),
        "WIP": wip(analyses, parts),
        "CER": cer(analyses, parts),
    }

# -----------------------
# Core evaluation on a dataset
# -----------------------

def evaluate_morph_dataset(dataset, tokenizer, topN_affixes: int = 200) -> Dict[str, Any]:
    words = dataset["word"]
    lemmas = dataset["lemma"]
    suffixes_list = dataset["suffixes"]

    # Collectors
    micro_tp = micro_fp = micro_fn = 0
    macro_prec = macro_rec = macro_f1 = 0.0
    macro_count = 0

    total_words = 0
    exact_analysis_match = 0

    analyses_seq = []
    parts_seq = []

    total_subwords = 0
    total_gold_morphs = 0
    total_pred_subwords = 0  # same as total_subwords

    over_seg_sum = 0.0
    under_seg_sum = 0.0

    lemma_single_hits = 0
    lemma_boundary_hits = 0

    # For affix coverage/atomicity
    suffix_counter = Counter()
    suffix_occurrences = defaultdict(int)
    suffix_atomic_hits = defaultdict(int)

    for word, lemma, suffixes in zip(words, lemmas, suffixes_list):
        total_words += 1

        gold_morphs, gold_b = gold_morpheme_segments(lemma, suffixes, word)
        pred_pieces, pred_b = predicted_boundaries(tokenizer, word)

        # Boundary P/R/F1 per item
        p, r, f = boundary_prf(gold_b, pred_b)
        macro_prec += p
        macro_rec += r
        macro_f1 += f
        macro_count += 1

        # Micro counts
        gs, ps = set(gold_b), set(pred_b)
        tp = len(gs & ps)
        fp = len(ps - gs)
        fn = len(gs - ps)
        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        # Exact analysis match sequence-level (for WER-like metrics we build strings)
        # analysis string uses '+' between morphemes; predicted uses '+' between pieces
        analysis_str = "+".join(gold_morphs)
        parts_str = "+".join(pred_pieces)
        analyses_seq.append(analysis_str)
        parts_seq.append(parts_str)
        if gold_morphs == pred_pieces:
            exact_analysis_match += 1

        # Length stats / over-under segmentation
        gm = len(gold_morphs)
        pm = len(pred_pieces)
        total_gold_morphs += gm
        total_pred_subwords += pm
        total_subwords += pm

        # Over-seg: subwords per morph; Under-seg: morphs per subword
        # Avoid division by zero
        if gm > 0 and pm > 0:
            over_seg_sum += pm / gm
            under_seg_sum += gm / pm

        # Lemma integrity signals
        lemma_single_hits += lemma_single_token(lemma, tokenizer)
        lemma_boundary_hits += lemma_boundary_hit(word, lemma, tokenizer)

        # Affix stats
        suffixes_list_split = suffixes.split("+") if suffixes else []
        for s in suffixes_list_split:
            suffix_counter[s] += 1
            suffix_occurrences[s] += 1
            # Atomic if the suffix appears as a whole predicted subword aligned with its span
            # We approximate by checking if s appears as a standalone piece anywhere in predicted pieces.
            if s in pred_pieces:
                suffix_atomic_hits[s] += 1

    # Aggregate boundary metrics
    micro_prec = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
    micro_rec = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
    micro_f1 = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) else 0.0

    macro_prec /= max(macro_count, 1)
    macro_rec /= max(macro_count, 1)
    macro_f1 /= max(macro_count, 1)

    # Sequence metrics
    ji = jiwer_seq_metrics(analyses_seq, parts_seq)

    # Over/under averages
    over_seg = over_seg_sum / max(total_words, 1)
    under_seg = under_seg_sum / max(total_words, 1)

    # Affix coverage/atomicity over top-N frequent suffixes
    top_suffixes = [s for s, _ in suffix_counter.most_common(topN_affixes)]
    # Coverage: how many of these suffix types appear as standalone vocab entries?
    # Cheap proxy: if s appears as a piece in any word, we count it covered.
    covered = sum(1 for s in top_suffixes if suffix_atomic_hits.get(s, 0) > 0)
    coverage = covered / max(len(top_suffixes), 1)

    # Atomicity: occurrences with standalone piece / total occurrences for these top suffixes
    at_hits = sum(suffix_atomic_hits.get(s, 0) for s in top_suffixes)
    at_total = sum(suffix_occurrences.get(s, 0) for s in top_suffixes)
    atomicity = at_hits / at_total if at_total else 0.0

    # Tokens-per-word
    mean_subwords_per_word = total_subwords / max(total_words, 1)

    # Lemma integrity
    lemma_single_rate = lemma_single_hits / max(total_words, 1)
    lemma_boundary_rate = lemma_boundary_hits / max(total_words, 1)

    # Exact sequence match rate
    exact_match_rate = exact_analysis_match / max(total_words, 1)

    return {
        "words": total_words,
        "mean_subwords_per_word": mean_subwords_per_word,
        "boundary_micro_P": micro_prec,
        "boundary_micro_R": micro_rec,
        "boundary_micro_F1": micro_f1,
        "boundary_macro_P": macro_prec,
        "boundary_macro_R": macro_rec,
        "boundary_macro_F1": macro_f1,
        "lemma_single_rate": lemma_single_rate,
        "lemma_boundary_rate": lemma_boundary_rate,
        "exact_morph_sequence_match": exact_match_rate,
        "over_seg": over_seg,
        "under_seg": under_seg,
        "affix_coverage_topN": coverage,
        "affix_atomicity_topN": atomicity,
        "WER": ji["WER"],
        "MER": ji["MER"],
        "WIL": ji["WIL"],
        "WIP": ji["WIP"],
        "CER": ji["CER"],
    }

# -----------------------
# Lemma-only evaluation (your existing helper)
# -----------------------

def evaluate_lemma_success(tokenizer, dataset):
    allw, matches = 0, 0
    single_lems = []
    for instance in dataset:
        lemma = instance["lemma"]
        allw += 1
        if lemma_single_token(lemma, tokenizer):
            single_lems.append(lemma)
            matches += 1
    prop = matches / allw if allw else 0.0
    return allw, matches, prop, single_lems

# -----------------------
# Orchestrator (per tokenizer config)
# -----------------------

def evaluate_tokenizer(tokenizer, out_dir, topN_affixes=200, dump_tokenizations=True):
    os.makedirs(out_dir, exist_ok=True)

    # Lemma-only sets
    allw, matches, prop, single_lemmas = evaluate_lemma_success(tokenizer, dset_lemmas)
    allw2, matches2, prop2, single_lemmas2 = evaluate_lemma_success(tokenizer, dset_common_lemmas)

    # Morph datasets
    stats_cekimli = evaluate_morph_dataset(dset_cekimli, tokenizer, topN_affixes)
    stats_nouns = evaluate_morph_dataset(dset_nouns, tokenizer, topN_affixes)
    stats_verbs = evaluate_morph_dataset(dset_verbs, tokenizer, topN_affixes)

    # Write lemma lists
    with open(os.path.join(out_dir, "single_lemmas.txt"), "w", encoding="utf-8") as f:
        for lemma in single_lemmas:
            f.write(lemma + "\n")
    with open(os.path.join(out_dir, "single_common_lemmas.txt"), "w", encoding="utf-8") as f:
        for lemma in single_lemmas2:
            f.write(lemma + "\n")

    # Optional: dump tokenizations for inspection (using cekimli set)
    if dump_tokenizations:
        with open(os.path.join(out_dir, "tokenizations_cekimli.txt"), "w", encoding="utf-8") as f:
            for w, lem, suf in zip(dset_cekimli["word"], dset_cekimli["lemma"], dset_cekimli["suffixes"]):
                gold_m, _ = gold_morpheme_segments(lem, suf, w)
                pred_p, _ = predicted_boundaries(tokenizer, w)
                f.write(f"{w}\tGOLD:{' + '.join(gold_m)}\tPRED:{' + '.join(pred_p)}\n")

    # Write stats summary
    with open(os.path.join(out_dir, "stats.txt"), "w", encoding="utf-8") as s:
        s.write("Lemma stats\n")
        s.write(f"All lemmas: {allw} matches: {matches} single-token rate: {prop:.4f}\n")
        s.write("Common lemma stats\n")
        s.write(f"All lemmas: {allw2} matches: {matches2} single-token rate: {prop2:.4f}\n")
        s.write("Morphological stats (Cekimli)\n")
        for k, v in stats_cekimli.items():
            s.write(f"{k}: {v}\n")
        s.write("###########\nMorphological stats (Common Nouns)\n")
        for k, v in stats_nouns.items():
            s.write(f"{k}: {v}\n")
        s.write("###########\nMorphological stats (Common Verbs)\n")
        for k, v in stats_verbs.items():
            s.write(f"{k}: {v}\n")

    return {
        "lemma_all": (allw, matches, prop),
        "lemma_common": (allw2, matches2, prop2),
        "cekimli": stats_cekimli,
        "nouns": stats_nouns,
        "verbs": stats_verbs,
    }

# -----------------------
# Batch evaluate a family
# -----------------------

def evaluate_tokenizer_family(collection_name, name_fmt="wordpiece_{}_cased_{}",
                              sizes=("2k","5k","10k","20k","32k","52k","128k"),
                              topN_affixes=200):
    for size in sizes:
        eval_dir = os.path.join(collection_name, size)
        os.makedirs(eval_dir, exist_ok=True)
        tokenizer_dir = name_fmt.format(size, collection_name)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        print(size, "started!")
        evaluate_tokenizer(tokenizer, eval_dir, topN_affixes=topN_affixes)
        print(size, "ended!")

# -----------------------
# Example: run multiple collections
# -----------------------

if __name__ == "__main__":
    tokenizer_dirs = ["books", "minimal", "alldata"]

    for tokenizer_dir in tokenizer_dirs:
        evaluate_tokenizer_family(tokenizer_dir)

