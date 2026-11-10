import math
import os
import tempfile
import time
import unittest

from evaluate import precision_recall
from search import Index, bm25_scores, idf, search, tokenize


def make_index(docs):
    tmp = tempfile.TemporaryDirectory()
    for name, text in docs.items():
        with open(os.path.join(tmp.name, name), "w") as f:
            f.write(text)
    index = Index(tmp.name)
    index.refresh()
    return tmp, index


class IndexTests(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("The Monsoon, in INDIA!"), ["monsoon", "india"])

    def test_postings_and_lengths(self):
        tmp, index = make_index({"a.txt": "rain rain sun", "b.txt": "sun"})
        self.assertEqual(index.postings["rain"], {"a.txt": 2})
        self.assertEqual(index.postings["sun"], {"a.txt": 1, "b.txt": 1})
        self.assertEqual(index.doc_len, {"a.txt": 3, "b.txt": 1})
        tmp.cleanup()

    def test_refresh_handles_new_changed_and_deleted_files(self):
        tmp, index = make_index({"a.txt": "rain", "b.txt": "sun"})
        time.sleep(0.01)
        with open(os.path.join(tmp.name, "a.txt"), "w") as f:
            f.write("snow")
        os.utime(os.path.join(tmp.name, "a.txt"), (time.time() + 5, time.time() + 5))
        os.remove(os.path.join(tmp.name, "b.txt"))
        with open(os.path.join(tmp.name, "c.txt"), "w") as f:
            f.write("wind")
        index.refresh()
        self.assertNotIn("rain", index.postings)
        self.assertNotIn("sun", index.postings)
        self.assertEqual(set(index.doc_len), {"a.txt", "c.txt"})
        self.assertEqual(index.refresh(), 0)  # nothing changed the second time
        tmp.cleanup()


class BM25Tests(unittest.TestCase):
    def test_matches_hand_calculation(self):
        tmp, index = make_index({"a.txt": "rain rain sun", "b.txt": "sun moon", "c.txt": "moon"})
        # N=3, df(rain)=1, |a|=3, avgdl=(3+2+1)/3=2
        expected_idf = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
        self.assertAlmostEqual(idf(index, "rain"), expected_idf)
        norm = 1.5 * (1 - 0.75 + 0.75 * 3 / 2)
        expected = expected_idf * 2 * 2.5 / (2 + norm)
        self.assertAlmostEqual(bm25_scores(index, "rain")["a.txt"], expected)
        tmp.cleanup()

    def test_rare_terms_count_more_and_ranking_order(self):
        tmp, index = make_index({"a.txt": "sun moon", "b.txt": "sun", "c.txt": "sun star"})
        results = search(index, "sun moon")
        self.assertEqual(results[0][0], "a.txt")
        tmp.cleanup()

    def test_unknown_words_give_no_results(self):
        tmp, index = make_index({"a.txt": "sun"})
        self.assertEqual(search(index, "zebra"), [])
        tmp.cleanup()


class MetricTests(unittest.TestCase):
    def test_precision_recall(self):
        results = [("a", 1), ("b", 1), ("c", 1)]
        self.assertEqual(precision_recall(results, ["a", "x"], 3), (1 / 3, 0.5))


if __name__ == "__main__":
    unittest.main()
