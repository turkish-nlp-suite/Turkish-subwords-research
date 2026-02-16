For subword-level vocab, we decompose each word into subwords via morhology tools eg `gittim` -> `git ##ti ##m`. Consequently, when we wanna cover some fraction of the words, we nned to know if all the subwords are present in the chosen vocabulary. Hence we introduce:


* k: size of the subword vocabulary taken from train (top-K most frequent subwords).
* token_coverage: fraction of word tokens in the eval split whose words are fully reconstructable from those top-K subwords (frequency-weighted).
* type_coverage: fraction of unique words (types) in the eval split that are fully reconstructable from those top-K subwords (unweighted).

"Fully reconstructable" again means every subword for the word appears in the top-K list.

Let's make an example. Let's say these are all the subwords, sorted by frequency top down:

```
-di
-ti
-me
-ma
-m  -> K=5
.
.
.
gel -> K=100
git -> K=101
.
.
.
giriş -> K=200
```

Let's say we have 6 words we wanna cover: `gittim, gitmedim, geldim, gelmedim, giriştim, girişmedim`. With K=5 we cannot cover any words, because first 5 subwords are only suffixes. With K=100 we can cover 2 words: `geldim, gelmedim` 33% of the words. With K=101 we cover 4 words: `gittim, gitmedim, geldim, gelmedim`, hence 66% of the words. For covering all the words we need to make K=200 to cover the root `giriş`.

---
Desomcposing all the corpus words take some time with morphology tools. Consequently we built a cache, where decomposition of each word is included eg `{"gittim": "git+ti+m, ..}"` as `wcache_{train|test|valid}.json`. Together word counts files as `{train|test|valid}_{word|subwrd}_counts.txt`, putting all these together we can calculate token and type coverages.
