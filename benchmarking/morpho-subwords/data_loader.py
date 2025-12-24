from torch.utils.data import Dataset, DataLoader
import torch
import string, json

def kill_punct(text):
  punct = string.punctuation
  text = text.translate(text.maketrans(' ', ' ', punct))
  text = " ".join(text.strip().split())
  return text



class Vocab:
  def __init__(self, subword_vocab_json, word_cache_file):
    self.vocab_json = subword_vocab_json
    self.cache_json = word_cache_file

    self.vocab_dict = self.make_vocab_to_ids_dict()
    self.word_cache = self.load_cache()

  def vocab_size(self):
    return len(self.vocab_dict)

  def make_vocab_to_ids_dict(self):
    with open(self.vocab_json, "r") as infile:
      vocab_lines = infile.read().split("\n")
    vocab_lines = list(filter(None, vocab_lines))
    vocab_lines = ["PAD", "UNK", " "] + vocab_lines
    vocab_dict = {subwrd:index for index,subwrd in enumerate(vocab_lines)}
    return vocab_dict

  def load_cache(self):
    with open(self.cache_json, "r") as infile:
      cache = json.load(infile)
    return cache

  def tokenize(self, word):
    subwrds = self.word_cache.get(word, [word])
    return subwrds

  def subwrd_to_id(self, subwrd):
    unk_id = 1
    return self.vocab_dict.get(subwrd, unk_id)

  def word_to_ids(self, word):
    subwrds = self.tokenize(word)
    ids = [self.subwrd_to_id(subwrd) for subwrd in subwrds]
    return ids

  def text_to_ids(self, text):
    text = text.lower()
    text = kill_punct(text)
    words =  text.split()
    id_list = [self.word_to_ids(word) for word in words]
    id_list = [item for row in id_list for item in row] # flatten
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



class SubWordCollator:
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
   

def subword_loader(dataset, vocab_obj, max_len, batch_size, shuffle=False):
  collator = SubWordCollator(vocab_obj, max_len)
  return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)
  
'''
vocab_obj = Vocab("vocab/movies/all_vocab.txt", "vocab/movies/wcache_movies.json")
x = vocab_obj.word_to_ids("gittim")
y=vocab_obj.tokenize("gittim")
print(x, y)
ch_collator = CharCollator(vocab_obj, 10)
batch = {
"sentence": ["dsjd", "shdhg dhdgdf", "sdhfhhfh fddfhfh"],
"star": [1, 2, 3]
}

x = ch_collator(batch)
print(x)
'''

