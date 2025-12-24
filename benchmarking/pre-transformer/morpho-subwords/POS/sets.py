import conllu
from torch.utils.data import Dataset


class UDSet(Dataset):
  def __init__(self, split):
    self.file = "tr_imst-ud-{}.conllu".format(split)
    data = open(self.file, "r")
    annotations = data.read()

    sentences = conllu.parse(annotations)

    all_tokens = []
    all_tags = []

    for sentence in sentences:
      tokens = []
      tags = []
      for token in sentence:
        tokent = token["form"]
        tag = token["upos"]
        tokens.append(tokent)
        tags.append(tag)
        all_tokens.append(tokens)
        all_tags.append(tags)
    self.sentences = all_tokens
    self.tags = all_tags

  def __len__(self):
    return len(self.sentences)


  def __getitem__(self, index):
    tags = self.tags[index]
    sentence = self.sentences[index]
    return {"tokens": sentence, "tags": tags}


