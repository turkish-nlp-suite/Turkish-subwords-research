from datasets import load_dataset
from torch.utils.data import DataLoader

from collections import Counter
import string, json, os

#task_names = ["movies", "e-commerce"]



task_to_keys = {
    "cola": "sentence",
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "sst2": "sentence", 
    "stsb": ("sentence1", "sentence2"),
}




def kill_punct(text):
  punct = string.punctuation
  text = text.translate(text.maketrans(' ', ' ', punct))
  text = " ".join(text.strip().split())
  return text


def make_vocab(taskname, split_name):
  ds = load_dataset("BayanDuygu/TrGLUE", taskname)
  all_words = Counter()

  dataset = ds[split_name]
  data_loader = DataLoader(dataset, batch_size=1000)
  keys = task_to_keys[taskname]
  for batch in data_loader:
    if len(keys) ==2:
      texts1 = batch[keys[0]]
      texts2 = batch[keys[1]]
      texts = texts1 + texts2
    else:
      texts = batch[keys]
    all_texts  = " ".join(texts).lower()
    all_texts = kill_punct(all_texts)
    words = all_texts.split()
    all_words.update(words)



  lenc = len(all_words)
  ordered_pairs = all_words.most_common(lenc)
  with open(taskname+ "/" + split_name + "_word_counts_" + taskname + ".txt", "w") as ofile:
    for wrd,count in ordered_pairs:
      ofile.write(wrd + " " + str(count) + "\n")


task_names = ["cola", "sst2", "mnli", "mrpc", "stsb"]
split_all = ["train", "test", "validation"]
split_mnli = ["train", "validation_matched", "validation_mismatched", "test_matched", "test_mismatched"]

for task in task_names:
  os.makedirs(task, exist_ok=True)
  split_names = split_mnli if task=="mnli" else split_all
  for split in split_names:
    make_vocab(task, split)
