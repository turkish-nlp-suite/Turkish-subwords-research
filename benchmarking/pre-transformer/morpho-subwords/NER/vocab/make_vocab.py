from datasets import load_dataset
from torch.utils.data import DataLoader

import zeyrek
from collections import Counter

import json, re, string

analyzer = zeyrek.MorphAnalyzer()
puncts = string.punctuation

def parse_format(string):
  parts = string.split(":")
  newp = []
  for part in parts:
    print(part, "part")
    if "+" in part or "|" in part:
      newparts = re.split(r"[:+|]", part)
      rp  = [p for p in newparts if p[0].islower()]
      newp += rp
    elif not part:
        pass
    elif part[0].isupper():
      pass
    else:
      newp.append(part)
  return newp


def get_subwords(word):
  if word in puncts:
    return [word]
  result = analyzer.analyze(word)
  if not result:
    return []
  word_result = result[0]
  w = word_result[0]
  if w.formatted == "Unk":
    return []
  _, formatted = w.formatted.split()
  newp = parse_format(formatted)
  return newp



def break_into(words):
  words2subwords = {}
  all_subwords = []
  for word in words:
    if word not in all_subwords:
      subwords  = get_subwords(word)
      if subwords:
        subwords = [subwords[0]] + ["##"+subwrd for subwrd in subwords[1:]]
        words2subwords[word] =  subwords
        all_subwords += subwords
  return all_subwords, words2subwords

def make_vocab():
  all_subwords = Counter()
  words_cache = {}
  dataset =  load_dataset("turkish-nlp-suite/turkish-wikiNER", split="train")
  for batch in dataset:
    batch_tokens = batch["tokens"]
    subwords, w2sbwrds = break_into(batch_tokens)
    all_subwords.update(subwords)
    words_cache.update(w2sbwrds)


  wcache = json.dumps(words_cache, ensure_ascii=False)
  with open("wcache_train.json", "w") as ofile:
    ofile.write(wcache)

  lenc = len(all_subwords)
  ordered_pairs = all_subwords.most_common(lenc)
  with open("train_subwrd_counts.json", "w") as ofile:
    for subwrd,count in ordered_pairs:
      ofile.write(subwrd + " " + str(count) + "\n")



make_vocab()
