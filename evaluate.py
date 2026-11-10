"""Precision@k and recall@k for every query in queries.json, and the averages."""

import json
import sys

from search import Index, search


def precision_recall(results, relevant, k):
    top = [doc for doc, _ in results[:k]]
    hits = len(set(top) & set(relevant))
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.0
    return precision, recall


def main(k=3):
    index = Index("docs")
    index.refresh()
    queries = json.load(open("queries.json"))
    total_p = total_r = 0.0
    print(f"{'query':<30}{'P@' + str(k):>6}{'R@' + str(k):>6}  top results")
    for q in queries:
        results = search(index, q["query"], k)
        p, r = precision_recall(results, q["relevant"], k)
        total_p, total_r = total_p + p, total_r + r
        print(f"{q['query']:<30}{p:>6.2f}{r:>6.2f}  {', '.join(d for d, _ in results)}")
    n = len(queries)
    print(f"\nmean precision@{k} = {total_p / n:.3f}, mean recall@{k} = {total_r / n:.3f} over {n} queries")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
