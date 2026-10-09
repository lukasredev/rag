"""Starter pipeline for the `codesearch` track -- runs end to end, deliberately weak.

It indexes only the FIRST LINE of every function (the `def ...(...)` signature) with BM25 and
returns the 10 best-scoring function ids. Most of the information in a function is thrown away.

Run it:
    python -m ragkit.run --track codesearch --pipeline tracks/codesearch/skeleton/pipeline.py --limit 20
"""

import json
import pathlib

from ragkit import trace  # only records anything inside RAG Studio
from ragkit.bm25 import BM25
from ragkit.text import tokenize


class Pipeline:
    def __init__(self, data_dir: pathlib.Path):
        with open(data_dir / "corpus.jsonl") as f:
            self.functions = [json.loads(line) for line in f]
        # TODO: index more than the signature line (the body! the file path!).
        # TODO: `tokenize` lowercases and splits on non-alphanumerics -- what happens to camelCase names?
        docs = [tokenize(fn["code"].split("\n", 1)[0]) for fn in self.functions]
        self.bm25 = BM25(docs)

    def answer(self, question: dict) -> dict:
        # TODO: the question is natural language, the corpus is code. How could you bridge that gap?
        #       (dense embeddings, query rewriting with an LLM, hypothetical code, reranking...)
        hits = self.bm25.search(tokenize(question["question"]), k=10)
        ids = [self.functions[i]["id"] for i, _ in hits]
        trace.log("retrieve", "BM25 over signatures", hits=[{"id": self.functions[i]["id"], "score": s, "text": self.functions[i]["code"][:600]} for i, s in hits])
        return {"answer": ids[0] if ids else "", "citations": ids}
