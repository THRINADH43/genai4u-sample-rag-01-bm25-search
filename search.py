"""BM25 search over a folder of .txt files, written from the formula. Standard library only.

python search.py docs "monsoon rain"     # top results
python evaluate.py                      # precision and recall on queries.json
"""

import math
import os
import re
import sys
from collections import Counter, defaultdict

STOPWORDS = {"a", "an", "the", "is", "are", "of", "in", "on", "to", "and", "or", "for", "with",
             "it", "by", "as", "at", "be", "can", "such", "every", "its", "how", "what", "which"}


def tokenize(text):
    """Lower case, split on anything that is not a letter or digit, drop stop words."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class Index:
    """Inverted index: term -> {doc_id: count}. Also keeps each document's length."""

    def __init__(self, folder):
        self.folder = folder
        self.postings = defaultdict(dict)
        self.doc_len = {}
        self.mtimes = {}

    def add(self, doc_id, text):
        self.remove(doc_id)
        counts = Counter(tokenize(text))
        for term, n in counts.items():
            self.postings[term][doc_id] = n
        self.doc_len[doc_id] = sum(counts.values())

    def remove(self, doc_id):
        if doc_id not in self.doc_len:
            return
        for term in list(self.postings):
            self.postings[term].pop(doc_id, None)
            if not self.postings[term]:
                del self.postings[term]
        del self.doc_len[doc_id]
        self.mtimes.pop(doc_id, None)

    def refresh(self):
        """Bring the index up to date with the folder: index new and changed files, drop
        deleted ones. Unchanged files (same modification time) are skipped."""
        on_disk = {}
        for name in os.listdir(self.folder):
            if name.endswith(".txt"):
                on_disk[name] = os.path.getmtime(os.path.join(self.folder, name))
        for doc_id in set(self.doc_len) - set(on_disk):
            self.remove(doc_id)
        changed = 0
        for doc_id, mtime in on_disk.items():
            if self.mtimes.get(doc_id) != mtime:
                with open(os.path.join(self.folder, doc_id), encoding="utf-8") as f:
                    self.add(doc_id, f.read())
                self.mtimes[doc_id] = mtime
                changed += 1
        return changed

    @property
    def n_docs(self):
        return len(self.doc_len)

    @property
    def avg_len(self):
        return sum(self.doc_len.values()) / self.n_docs if self.n_docs else 0.0


def idf(index, term):
    """BM25 idf with +1 inside the log so it is never negative:
    log((N - df + 0.5) / (df + 0.5) + 1)."""
    df = len(index.postings.get(term, {}))
    return math.log((index.n_docs - df + 0.5) / (df + 0.5) + 1)


def bm25_scores(index, query, k1=1.5, b=0.75):
    """score(D, Q) = sum over query terms of
       idf(t) * tf(t, D) * (k1 + 1) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))"""
    scores = defaultdict(float)
    for term in tokenize(query):
        weight = idf(index, term)
        for doc_id, tf in index.postings.get(term, {}).items():
            norm = k1 * (1 - b + b * index.doc_len[doc_id] / index.avg_len)
            scores[doc_id] += weight * tf * (k1 + 1) / (tf + norm)
    return scores


def search(index, query, k=5):
    scores = bm25_scores(index, query)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def main():
    folder, query = sys.argv[1], " ".join(sys.argv[2:])
    index = Index(folder)
    index.refresh()
    for doc_id, score in search(index, query):
        print(f"{score:6.3f}  {doc_id}")


if __name__ == "__main__":
    main()
