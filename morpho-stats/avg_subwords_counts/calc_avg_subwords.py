from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os
import string



oscard4 = load_dataset("turkish-nlp-suite/temiz-OSCAR-2019", split="train")
oscar_loader = DataLoader(oscard4, batch_size=1000)

kitaplar2 = load_dataset("turkish-nlp-suite/Kitaplar", split="train")
books_loader = DataLoader(kitaplar2, batch_size=1000)

acrawl2 = load_dataset("turkish-nlp-suite/Akademik-Ozetler", split="train")
acrawl_loader = DataLoader(acrawl2, batch_size=1000)

ccrawl2 = load_dataset("turkish-nlp-suite/KulturHaritasi", split="train")
crafted_loader = DataLoader(ccrawl2, batch_size=1000)

all_loaders =  [books_loader, acrawl_loader, oscar_loader, crafted_loader]




def calculate_count(tokenized_sentence, all_text):
  # fertility
  all_words = all_text.split()
  len_words = sum([len(word) for word in all_words])

  all_subwords = len(tokenized_sentence)
  tokenized_sentence = [subw.lstrip("##") for subw in tokenized_sentence]
  len_subw = sum([len(subw) for subw in tokenized_sentence])

  return all_subwords, len(all_words), len_words, len_subw 


def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def goover_dataset(loaders, tokenizer, finalnums):
  all_subwords, all_words, len_words, len_subwords = 0, 0, 0, 0
  index = 0
  for loader, finalnum in zip(loaders, finalnums):
    for batch in loader:
      if index >= finalnum:
          break
      texts = batch["text"]
      all_text =  " ".join(texts)
      all_text = remove_punctuation(all_text)
      tokenized_text = tokenizer.tokenize(all_text)
      alls, allw, lenwords, lensubw = calculate_count(tokenized_text, all_text) 

      all_subwords += alls
      all_words += allw
      len_words += lenwords
      len_subwords += lensubw
      index += 1

  print(all_subwords, all_words, len_words, len_subwords, "FIINAL")
  avgwrd = len_words / all_words
  avgsubwrd = len_subwords / all_subwords
  return all_words, all_subwords, avgwrd, avgsubwrd


def evaluate_tokenizer(tokenizer, tokenizer_dir):
  all_words, all_subwords, avgwrd, avgsubwrd = goover_dataset(all_loaders, tokenizer, [1000, 50, 50, 20])

  stats_file = tokenizer_dir + "/stats2.txt"

  with open(stats_file, "w") as sfile:
    sfile.write("Word stats\n")
    sfile.write("Word count: " + str(all_words) + " Average word length: " + str(avgwrd) + " Subword count: " + str(all_subwords) + " Average subword length : " + str(avgsubwrd))
    sfile.write("\n")


def evaluate_tokenizer_family(collection_name):
  sizes = ["2k", "5k", "10k", "20k", "32k", "52k", "128k"]
  name = "turkish-nlp-suite/wordpiece_{}_cased_{}"

  for size in sizes:
    eval_dir  = collection_name + "/" + size
    tokenizer_dir  =  name.format(size, collection_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print(size, " started!")
    evaluate_tokenizer(tokenizer, eval_dir)
    print(size, " ended!")



tokenizer_dirs = ["books", "oldbooks", "minimal", "alldata"]


for tokenizer_dir in tokenizer_dirs:
  print("collection started ", tokenizer_dir)
  evaluate_tokenizer_family(tokenizer_dir)

