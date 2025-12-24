import json



def take_subwords(number, all_subwords):
  all_subwords = all_subwords[:number]
  return all_subwords


def is_word_fully_covered(morphs, train_subwords):
  return all([morph in train_subwords for morph in morphs])
  
def is_word_partially_covered(morphs, train_subwords):
  return any([morph in train_subwords for morph in morphs])


def calculate_coverage(train_subwords, word_counts, testing_subwords):
  total_words, partially_covered_words, fully_covered_words, unique_words = 0, 0, 0, 0
  for word,morphs in testing_subwords.items():
    count = word_counts.get(word, 0)
    if count == 0:
      continue
    total_words += count
    unique_words += 1
    fully_covered = is_word_fully_covered(morphs, train_subwords)
    partially_covered = not fully_covered and is_word_partially_covered(morphs, train_subwords)
    fully_covered_words += count if fully_covered else 0
    partially_covered_words += count if partially_covered else 0
  return unique_words, total_words, fully_covered_words, partially_covered_words



def calculate_task_size(split, N):
  train_file = "train_vocab.txt"
  train_subwords = open(train_file).read().split("\n")
  train_subwords = list(filter(None, train_subwords))

  if N != "all":
      train_subwords = train_subwords[:N]

  counts_file = "{}_word_counts.txt".format(split)
  word_counts = {}

  with open(counts_file, "r") as infile:
    for line in infile:
      #print(line.strip())
      wrd, cnt = line.strip().split()
      word_counts[wrd] =  int(cnt)

  subwrds_file = "{}_wcache.json".format(split)
  with open(subwrds_file, "r") as infile:
    testing_subwords = json.load(infile)
  uwords, twords, cwords, pwords  = calculate_coverage(train_subwords, word_counts, testing_subwords)
  lefties = twords-cwords-pwords
  print("unique words: ", uwords, "total words: ", twords, "fully covered words: ", cwords, "partically covered: ", pwords, "not covered: ", lefties, cwords/twords, pwords/twords, lefties/twords)
  print(N, "vocab size")
  print("=============")


def calculate_all():
  #tasks = ["movies", "e-commerce"]
  sizes =  [50, 100, 250, 500, 750, 1000, 2000, 3000, 4000, 5000, 7500, 9000, 10000, 15000, "all"]
  splits = ["train", "valid", "test"]

  for split in splits:
      print("SPLIT", split, "=========================")
      for size in sizes:
        calculate_task_size(split, size)



calculate_all()
