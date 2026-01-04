import json


def fetch_subwords_from_cache(word, cache):
  if not word: return word
  subwords = cache.get(word, [word])
  if word[0].isupper():
      fw = subwords[0]
      fw = fw[0].upper() + fw[1:]
      subwords = [fw] + subwords[1:]
  return subwords





with open("vocab_files/wcache_valid.json", "r") as infile:
    cache = json.load(infile)

word = "Harbin"

sr = fetch_subwords_from_cache(word, cache)

print(sr)

#result ['Harb', '##in']
