import json








def calculate_avg_suff(task):
  train_file = "{}/train_word_counts_{}.txt".format(task, task)
  train_words = {}

  with open(train_file, "r") as infile:
    for line in infile:
      #print(line.strip())
      wrd, cnt = line.strip().split()
      train_words[wrd] =  int(cnt)

  cache_file = "{}/words_cache.json".format(task)
  with open(cache_file, "r") as infile:
    word_cache = json.load(infile)

  num_all_wrds = 0
  allsuffs = 0
  for word,cnt in train_words.items():
    if word in word_cache:
      subwrds = word_cache[word] 
      num_all_wrds += cnt
      num_suffs = len(subwrds)-1
      total_suffs = cnt * num_suffs
      allsuffs += total_suffs
  print(num_all_wrds)

  print(allsuffs/num_all_wrds)



calculate_avg_suff("movies")
calculate_avg_suff("e-commerce")

