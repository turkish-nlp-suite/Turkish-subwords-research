For subword-level vocab, we decompose each word into subwords via morhology tools eg `gittim` -> `git ##ti ##m`. Consequently, when we wanna cover some fraction of the words, we nned to know if all the subwords are present in the chosen vocabulary. Hence we introduce:


* k: size of the subword vocabulary taken from train (top-K most frequent subwords).
* token_coverage: fraction of word tokens in the eval split whose words are fully reconstructable from those top-K subwords (frequency-weighted).
* type_coverage: fraction of unique words (types) in the eval split that are fully reconstructable from those top-K subwords (unweighted).

"Fully reconstructable: again means every subword for the word appears in the top-K list.

Desomcposing all the corpus words take some time. Consequently we built a cache, where decomposition of each word is included eg `{"gittim": "git+ti+m, ..}"` as `wcache_{train|test|valid}.json`. Together word counts files as `{train|test|valid}_{word|subwrd}_counts.txt`, 
