from datasets import load_dataset


dataset = load_dataset('turkish-nlp-suite/turkish-wikiNER', split='validation+train+test')

all_len = 0

for instance in dataset:
     lent = len(instance["tokens"])
     all_len = max(lent, all_len)
     print(lent)



