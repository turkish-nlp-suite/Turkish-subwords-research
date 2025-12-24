from datasets import load_dataset
from torch.utils.data import DataLoader

from collections import Counter
import string, json

#task_names = ["movies", "e-commerce"]
task_names = ["e-commerce"]
split_names = ["train", "test", "validation"]

ds = load_dataset("maydogan/TRSAv1", split="train")

ds_train_devtest = ds.train_test_split(test_size=0.2, seed=42)
ds_devtest = ds_train_devtest['test'].train_test_split(test_size=0.5, seed=42)


sets = {
    'train': ds_train_devtest['train'],
    'validation': ds_devtest['train'],
    'test': ds_devtest['test']
}



def kill_punct(text):
  punct = string.punctuation
  text = text.translate(text.maketrans(' ', ' ', punct))
  text = " ".join(text.strip().split())
  return text


def make_vocab(taskname, split_name):
  all_words = Counter()

  dataset = sets[split_name]
  data_loader = DataLoader(dataset, batch_size=1000)
  for batch in data_loader:
    texts = batch["review"]
    all_texts  = " ".join(texts).lower()
    all_texts = kill_punct(all_texts)
    words = all_texts.split()
    all_words.update(words)



  lenc = len(all_words)
  ordered_pairs = all_words.most_common(lenc)
  with open(taskname+ "/" + split_name + "_word_counts_" + taskname + ".txt", "w") as ofile:
    for wrd,count in ordered_pairs:
      ofile.write(wrd + " " + str(count) + "\n")



for task in task_names:
  for split in split_names:
    make_vocab(task, split)

