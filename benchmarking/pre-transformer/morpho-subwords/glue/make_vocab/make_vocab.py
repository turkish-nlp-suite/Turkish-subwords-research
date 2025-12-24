from datasets import load_dataset
from torch.utils.data import DataLoader

import zeyrek
from collections import Counter

import json, re, string

analyzer = zeyrek.MorphAnalyzer()

def parse_format(string):
  parts = string.split(":")
  newp = []
  for part in parts:
    if "+" in part or "|" in part:
      newparts = re.split(r"[:+|]", part)
      rp  = [p for p in newparts if p[0].islower()]
      newp += rp
    elif part[0].isupper():
      pass
    else:
      newp.append(part)
  return newp


def get_subwords(word):
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




def kill_punct(text):
  punct = string.punctuation
  text = text.translate(text.maketrans(' ', ' ', punct))
  text = " ".join(text.strip().split())
  return text


def break_into(text):
  words2subwords = {}
  all_subwords = []
  words = text.split()
  for word in words:
    if word not in all_subwords:
      subwords  = get_subwords(word)
      if subwords:
        subwords = [subwords[0]] + ["##"+subwrd for subwrd in subwords[1:]]
        words2subwords[word] =  subwords
        all_subwords += subwords
  return all_subwords, words2subwords

def make_vocab(taskname):
  all_subwords = Counter()
  words_cache = {}
  #dataset =  load_dataset("turkish-nlp-suite/SentiTurca", taskname, split="train")
  ds = load_dataset("maydogan/TRSAv1", split="train")

  ds_train_devtest = ds.train_test_split(test_size=0.2, seed=42)
  #ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=42)


  dataset =  ds_train_devtest['train']
  data_loader = DataLoader(dataset, batch_size=1000)
  for batch in data_loader:
    texts = batch["review"]
    all_texts  = " ".join(texts).lower()
    all_texts = kill_punct(all_texts)
    subwords, w2sbwrds = break_into(all_texts)
    all_subwords.update(subwords)
    words_cache.update(w2sbwrds)


  wcache = json.dumps(words_cache, ensure_ascii=False)
  with open("wcache_" + taskname + ".json", "w") as ofile:
    ofile.write(wcache)

  lenc = len(all_subwords)
  ordered_pairs = all_subwords.most_common(lenc)
  with open("subwrd_counts_" + taskname + ".json", "w") as ofile:
    for subwrd,count in ordered_pairs:
      ofile.write(subwrd + " " + str(count) + "\n")



make_vocab("e-commerce")
#make_vocab("movies")
