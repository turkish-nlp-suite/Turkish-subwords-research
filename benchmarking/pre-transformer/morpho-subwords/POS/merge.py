import json

with open("wcache_test.json", "r") as file1, open("wcache_valid.json", "r") as file2, open("wcache_train.json", "r") as file3, open("words_cache.json", "w") as ofile:
    js1 = json.load(file1)
    js2 = json.load(file2)
    js3 = json.load(file3)
    js1.update(js2)
    js1.update(js3)
    bgjs = json.dumps(js1, ensure_ascii=False)
    ofile.write(bgjs)

