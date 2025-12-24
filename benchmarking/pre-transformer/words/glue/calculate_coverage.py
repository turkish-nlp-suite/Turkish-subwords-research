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


def calculate_task_size(task, size, split):
  testing_file = "{}/{}_word_counts_{}.txt".format(task, split, task)
  testing_word_counts = []
  with open(testing_file, "r") as infile:
    for line in infile:
      wrd, cnt = line.strip().split()
      testing_word_counts.append((wrd, int(cnt)))
  vocab = []
  vocab_file = "{}/vocab_train.txt".format(task, task, size)
  with open(vocab_file, "r") as infile:
    for line in infile:
      word = line.strip()
      vocab.append(word)
  if size> 0:
    vocab = vocab[:size]
  vocab = set(vocab)
  uwords, twords, cwords = calculate_coverage(vocab, testing_word_counts)
  print(uwords, twords, cwords, cwords/twords)
  print(size)
  print("=============")


def calculate_all():
  tasks = ["movies", "e-commerce"]
  #tasks = ["e-commerce"]
  #sizes =  [50, 100, 200, 500, 800, 1000, 1500, 2000, 3000, 5000, 10000, 15000, 20000, 30000, 50000, 80000, 100000, 120000, 150000, 200000, 250000, 0]
  sizes =  [50, 100, 200, 500, 800, 1000, 1500, 2000, 3000, 5000, 10000, 15000, 20000, 30000, 50000, 80000, 0]
  splits = ["train", "validation", "test"]

  for task in tasks:
    print("TASK ", task , "=====================================================")
    for split in splits:
      print("SPLIT", split, "=========================")
      for size in sizes:
        calculate_task_size(task, size, split)
      


calculate_all()

