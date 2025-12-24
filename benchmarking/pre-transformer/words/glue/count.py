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


def make_vocab(taskname):
  all_words = Counter()

  data_loader = DataLoader(ds, batch_size=1000)
  for batch in data_loader:
    texts = batch["review"]
    for text in texts:
      text = kill_punct(text)
      words = text.split()
      lenc = len(words)
      print(lenc)




for task in task_names:
    make_vocab(task)

