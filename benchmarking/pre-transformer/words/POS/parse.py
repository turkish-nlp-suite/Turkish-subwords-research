import conllu

data_file = "tr_imst-ud-test.conllu"

data = open(data_file, "r")
annotations = data.read()

sentences = conllu.parse(annotations)

all_tokens = []
all_tags = []

cumm = 0
tt = []
for sentence in sentences:
  tokens = []
  tags = []
  for token in sentence:
    #print(token["form"], token["upos"])
    tokent = token["form"]
    tag = token["upos"]
    tokens.append(tokent)
    tags.append(tag)
    tt.append(tokent)
  all_tokens.append(tokens)
  #all_tags.append(tags)


from collections import Counter
cnts = Counter(tt)
lens = len(cnts)

mostp= cnts.most_common(lens)



with open("test_words_count.txt", "w") as ofile:
  for tag,cnt in mostp:
    ofile.write(tag + " " + str(cnt) + "\n")


