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





def calculate_fertility(tokenized_sentence):
  # fertility
  word_start_mask = [0 if token.startswith("##") else 1 for token in tokenized_sentence] # number of language words
  # Ben de git ##ti #m oraya -> 1 1 1 0 0 1 
  all_subwords = len(word_start_mask)
  all_words = sum(word_start_mask)  # fertility is ratio subwords / all_words

  # proprotion of split words and not split words
  mask_pairs = zip(word_start_mask, word_start_mask[1:])
  non_split_words = len([(m1,m2) for (m1,m2) in mask_pairs if m1 == 1 and m2 != 0])
  if word_start_mask[-1] == 1:
    non_split_words += 1
  split_words = all_words - non_split_words # single token rate is non_split_words  / all_words and continued rate is split_words / all_words

  return all_subwords, all_words, split_words, non_split_words


def remove_punctuation(text):
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)

def goover_dataset(loader, tokenizer, finalnum):
  all_subwords, all_words, split_words, non_split_words = 0, 0, 0, 0
  index = 0
  for batch in loader:
      if index >= finalnum:
          break
      texts = batch["text"]
      all_text =  " ".join(texts)
      all_text = remove_punctuation(all_text)
      tokenized_text = tokenizer.tokenize(all_text)
      alls, allw, splitw, non_split_w = calculate_fertility(tokenized_text) 

      all_subwords += alls
      all_words += allw
      split_words += splitw
      non_split_words += non_split_w
      index += 1

  print(all_subwords, all_words, split_words, non_split_words, "FIINAL")
  fertility = all_subwords / all_words
  continued_words = split_words / all_words
  single_tokens = non_split_words / all_words
  return fertility, continued_words, single_tokens


def evaluate_tokenizer(tokenizer, tokenizer_dir):
  fer1, cont1, singles1 = goover_dataset(books_loader, tokenizer, 1000)
  print("books finished")
  fer2, cont2, singles2 = goover_dataset(acrawl_loader, tokenizer, 50)
  print("acrawl finished")
  fer3, cont3, singles3 = goover_dataset(oscar_loader, tokenizer, 50)
  print("oscar finished")
  fer4, cont4, singles4 = goover_dataset(crafted_loader, tokenizer, 20)
  print("ccrawl finished")

  stats_file = tokenizer_dir + "/stats.txt"

  with open(stats_file, "w") as sfile:
    sfile.write("Books stat\n")
    sfile.write("Fertility: " + str(fer1) + " Continued tokens: " + str(cont1) + " Single tokens: " + str(singles1))
    sfile.write("\n###########\n")
    sfile.write("Academic crawl stats\n")
    sfile.write("Fertility: " + str(fer2) + " Continued tokens: " + str(cont2) + " Single tokens: " + str(singles2))
    sfile.write("\n###########\n")
    sfile.write("Crafted crawl stats\n")
    sfile.write("Fertility: " + str(fer4) + " Continued tokens: " + str(cont4) + " Single tokens: " + str(singles4))
    sfile.write("OSCAR stats\n")
    sfile.write("Fertility: " + str(fer3) + " Continued tokens: " + str(cont3) + " Single tokens: " + str(singles3))
    sfile.write("\n")


def evaluate_tokenizer_family(collection_name):
  sizes = ["2k", "5k", "10k", "20k", "32k", "52k", "128k"]
  name = "turkish-nlp-suite/wordpiece_{}_cased_{}"

  for size in sizes:
    eval_dir  = collection_name + "/" + size
    try:
      os.mkdir(eval_dir)
    except:
      pass
    tokenizer_dir  =  tokenizer_basedir + "/" +  name.format(size)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    print(size, " started!")
    evaluate_tokenizer(tokenizer, eval_dir)
    print(size, " ended!")



tokenizer_dirs = ["books", "minimal", "alldata"]


for tokenizer_dir in tokenizer_dirs:
  print("collection started ", tokenizer_dir)
  evaluate_tokenizer_family(tokenizer_dir)

