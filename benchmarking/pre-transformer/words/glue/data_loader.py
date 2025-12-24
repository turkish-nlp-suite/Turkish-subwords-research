from torch.utils.data import Dataset, DataLoader
import torch
import string


def kill_punct(text):
  punct = string.punctuation
  text = text.translate(text.maketrans(' ', ' ', punct))
  text = " ".join(text.strip().split())
  return text



class Vocab:
  def __init__(self, vocab_file):
    self.vocab_file = vocab_file
    self.vocab_dict = self.make_vocab_to_ids_dict()

  def vocab_size(self):
    return len(self.vocab_dict)

  def make_vocab_to_ids_dict(self):
    vocab_lines = open(self.vocab_file, "r").read().split("\n")
    vocab_lines = list(filter(None, vocab_lines))

    vocab_lines = ["PAD", "UNK"] + vocab_lines
    vocab_dict = {word:index for index,word in enumerate(vocab_lines)}
    return vocab_dict


  def tokenize(self, text):
    text = text.lower()
    text = kill_punct(text)
    words = text.split()
    return words

  def word_to_id(self, word):
    unk_id = 1
    return self.vocab_dict.get(word, unk_id)

  def text_to_ids(self, text):
    tokens = self.tokenize(text)
    id_list = [self.word_to_id(token) for token in tokens]
    return id_list

  def texts_to_ids(self, texts):
    ids_list = []
    for text in texts:
      ids = self.text_to_ids(text)
      ids_list.append(ids)
    return ids_list

  def pad_to_length(self, ids_list, max_len):
    pad_id =0
    curr_len = len(ids_list)
    return ids_list + [pad_id] * (max_len - curr_len)

  def truncate(self, ids_list, max_len):
    curr_len = len(ids_list)
    diff_len = curr_len - max_len 
    # take the mid_part
    start_ind = diff_len // 2
    return ids_list[start_ind:start_ind+max_len]



  def pad_or_truncate(self, ids_list, max_len):
    curr_len = len(ids_list)
    if curr_len < max_len:
      ids_list = self.pad_to_length(ids_list, max_len)
    elif curr_len > max_len:
      ids_list = self.truncate(ids_list, max_len)
    return ids_list



class  WordCollator:
  def __init__(self, vocab_obj, max_len):
    self.vocab_obj = vocab_obj
    self.max_len = max_len

  def __call__(self, batch):
    #print(batch)
    texts = [item["sentence"] for item in batch]
    labels =[item["star"] for item in batch]
    ids_list = self.vocab_obj.texts_to_ids(texts) 
    running_max_len = self._locate_max_len(ids_list)
    ids_list = [self.vocab_obj.pad_or_truncate(id_list, self.max_len) for id_list in ids_list]
    return {"ids": torch.tensor(ids_list), "labels": torch.tensor(labels)}


  def _locate_max_len(self, ids_list):
    max_among = max([len(id_list) for id_list in ids_list])
    return max(max_among, self.max_len)
   

def word_loader(dataset, vocab_obj, max_len, batch_size, shuffle=False):
  collator = WordCollator(vocab_obj, max_len)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
  
'''
vocab_obj = Vocab("vocabs/movies/vocab_movies_2K.txt")
x = vocab_obj.word_to_id("sevdim")
x = vocab_obj.text_to_ids("sevdim")
print(x)
ch_collator = CharCollator(vocab_obj, 10)
batch = {
"sentence": ["dsjd", "shdhg dhdgdf", "sdhfhhfh fddfhfh"],
"star": [1, 2, 3]
}

x = ch_collator(batch)
print(x)
'''

