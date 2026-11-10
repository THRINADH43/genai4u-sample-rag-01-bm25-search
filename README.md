# BM25 search engine from scratch

A search engine in plain Python (no search library). It indexes a folder of `.txt` files
with an inverted index, ranks results with BM25 written from its formula, and measures
precision and recall on 16 test queries. No API key needed.

## Run

```
python search.py docs "monsoon rain"
python evaluate.py        # precision@3 and recall@3 (python evaluate.py 5 for k=5)
python -m unittest discover -s tests -t .
```

## Documents

`docs/` has 18 short documents I wrote (monsoon, cricket, UPI, ISRO, metro ...).

## How it works

- **Tokenizer:** lower case, split on non-letters and non-digits, drop common stop words.
- **Inverted index** (`Index`): `term -> {document: count}`, plus every document's length.
- **Folder updates:** `Index.refresh()` compares the folder with the index by file
  modification time. New and changed files are (re)indexed, deleted files are removed from
  every posting list, and unchanged files are skipped. Call it again any time the folder changes.
- **BM25** (`bm25_scores`), with k1 = 1.5 and b = 0.75:

  ```
  idf(t)     = log((N - df + 0.5) / (df + 0.5) + 1)
  score(D,Q) = sum over t in Q of  idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |D| / avgdl))
  ```

  The `+ 1` inside the log keeps idf positive for words found in more than half the
  documents. A unit test checks one score against a hand calculation.

## Results (k = 3)

```
mean precision@3 = 0.417, mean recall@3 = 1.000 over 16 queries
```

Recall is perfect: every relevant document is in the top 3 for every query. Precision looks
low only because most queries have one relevant document, so the best possible P@3 is 0.33
for them; my average best possible precision is 0.42, and I reach it. Extra results show up
because of shared words: "send money **from** phone" also returns `isro.txt` ("launch from")
since "from" is not in my stop word list, and "blood **sugar** control" also returns `tea.txt`
(masala tea has sugar). BM25 matches words, not meaning, so it cannot tell these apart.

## Tests

The tests cover tokenizing, posting lists, folder updates (new, changed and deleted files),
the BM25 formula against a hand calculation, ranking order and the metrics.
