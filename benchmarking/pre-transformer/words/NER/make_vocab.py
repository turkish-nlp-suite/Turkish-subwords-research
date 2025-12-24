from datasets import load_dataset


dataset = load_dataset('turkish-nlp-suite/turkish-wikiNER', split='validation')

all_tokens = []

for instance in dataset:
     all_tokens += instance["tokens"]


from collections import Counter
allt= Counter(all_tokens)
lent = len(allt)

pairs = allt.most_common(lent)

with open("valid_word_counts_movies.txt", "w") as ofile:
  for word,count in pairs:
      ofile.write(word + " " + str(count) + "\n")

