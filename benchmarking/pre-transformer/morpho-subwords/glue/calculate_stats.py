import json

'''
with open("movies/movies_words_cache.json") as infile:
  injs = json.load(infile)
  num_words, total_lens, total_subwrds = 0, 0, 0
  for word,split in injs.items():
    num_words += 1

    len_splits = len(split)
    sum_lens = sum([len(subwrd) for subwrd in split])

    total_subwrds += len_splits
    total_lens += sum_lens

  print(num_words, total_subwrds / num_words, total_lens/total_subwrds)

'''

from collections import Counter

with open("movies/movies_words_cache.json") as infile:
  injs = json.load(infile)
  all_splits = []
  all_lens = []
  for word,split in injs.items():

    len_splits = len(split)
    all_splits.append(len_splits)

    all_lens += [len(subwrd) for subwrd in split]

  all_splits = Counter(all_splits)
  all_lens = Counter(all_lens)

print(all_splits)
print(all_lens)
