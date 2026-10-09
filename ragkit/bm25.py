"""Okapi BM25 in ~50 lines, so you can see exactly how keyword search scores documents.

    from ragkit.bm25 import BM25
    from ragkit.text import tokenize
    index = BM25([tokenize(c) for c in chunks])
    hits = index.search(tokenize("how do I unzip a file"), k=5)   # [(chunk_index, score), ...]
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from . import trace


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.doc_len.mean()) if self.n_docs else 0.0
        # postings[term] = list of (doc_index, term_frequency)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in Counter(doc).items():
                self.postings[term].append((i, tf))
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()
        }

    def scores(self, query: list[str]) -> np.ndarray:
        """BM25 score of every document for this query (array of length n_docs)."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / max(self.avg_len, 1e-9))
        for term in set(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                scores[i] += idf * tf * (self.k1 + 1) / (tf + norm[i])
        return scores

    def search(self, query: list[str], k: int = 10) -> list[tuple[int, float]]:
        scores = self.scores(query)
        if self.n_docs == 0:
            return []
        k = min(k, self.n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        hits = [(int(i), float(scores[i])) for i in top if scores[i] > 0]
        if trace.active():
            terms = {t: round(self.idf[t], 2) if t in self.idf else None for t in dict.fromkeys(query)}
            trace.log("bm25", f"BM25 top {k}", terms=terms, hits=[{"index": i, "score": s} for i, s in hits])
        return hits
