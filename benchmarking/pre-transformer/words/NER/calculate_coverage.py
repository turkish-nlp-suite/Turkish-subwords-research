import json



def is_word_covered(word, vocab):
  return word in vocab


def calculate_coverage(vocab, testing_word_counts):
  total_words, covered_words = 0, 0
  unique_words = len(testing_word_counts)
  
  for word,count in testing_word_counts:
    total_words += count
    unique_words += 1
    covered = is_word_covered(word, vocab)
    covered_words += count if covered else 0
  return unique_words, total_words, covered_words


def calculate_task_size(split, N):
  testing_file = "{}_word_counts.txt".format(split)
  testing_word_counts = []
  with open(testing_file, "r") as infile:
    for line in infile:
      wrd, cnt = line.strip().split()
      testing_word_counts.append((wrd, int(cnt)))
  vocab = []
  vocab_file = "train_vocab.txt"
  with open(vocab_file, "r") as infile:
    for line in infile:
      word = line.strip()
      vocab.append(word)
  if N != "all":
      vocab = vocab[:N]
  vocab = set(vocab)
  uwords, twords, cwords = calculate_coverage(vocab, testing_word_counts)
  print(uwords, twords, cwords, cwords/twords)
  print(N)
  print("=============")


def calculate_all():
  sizes =  [50, 100, 200, 500, 750, 1000, 2000, 5000, 10000, 20000, 30000, 40000, 60000, "all"]
  splits = ["train", "valid", "test"]

  for split in splits:
      print("SPLIT", split, "=========================")
      for size in sizes:
        calculate_task_size(split, size)
      


calculate_all()

