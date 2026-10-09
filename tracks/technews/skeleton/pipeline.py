"""Starter pipeline for the Tech News Analyst track. It runs end to end, but it is deliberately weak.

What it does: cuts every article body into 1000-character windows, finds the 3 best windows with BM25
keyword search, and asks the LLM to answer from them. That's it.

Run it:
    RAG_ENV_FILE=... .venv/bin/python -m ragkit.run --track technews --pipeline tracks/technews/skeleton/pipeline.py --limit 20
"""

from __future__ import annotations

import json
import pathlib

from ragkit import trace  # only records anything inside RAG Studio
from ragkit.bm25 import BM25
from ragkit.llm import chat
from ragkit.text import chunk_chars, tokenize


class Pipeline:
    def __init__(self, data_dir: pathlib.Path):
        self.articles = [json.loads(line) for line in open(data_dir / "articles.jsonl")]
        # TODO: the articles have metadata (title, source, published_at, category, author).
        #       The questions talk about "The Verge", "TechCrunch", "an article published on October 7"...
        #       Right now none of that metadata is visible to the retriever or to the LLM.
        self.chunks, self.chunk_doc = [], []
        for a in self.articles:
            for piece in chunk_chars(a["body"], size=1000, overlap=100):  # TODO: is this a good chunk?
                self.chunks.append(piece)
                self.chunk_doc.append(a["id"])
        self.bm25 = BM25([tokenize(c) for c in self.chunks])
        # TODO: keyword search alone? Try dense embeddings (ragkit.embed) or a hybrid of both.

    def answer(self, question: dict) -> dict:
        q = question["question"]
        hits = self.bm25.search(tokenize(q), k=3)  # TODO: 3 chunks for a question that needs 2-4 articles?
        trace.log("retrieve", "BM25 chunks", hits=[{"id": self.chunk_doc[i], "score": s, "text": self.chunks[i]} for i, s in hits])
        context = "\n\n---\n\n".join(self.chunks[i] for i, _ in hits)
        # TODO: tell the model the answer format (Yes/No, a short name, or "Insufficient information.")
        #       and when it should abstain.
        prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"
        answer = chat(prompt, max_tokens=300)
        # TODO: which ids should you cite? (only the first 5 distinct ids count)
        citations = list(dict.fromkeys(self.chunk_doc[i] for i, _ in hits))
        return {"answer": answer, "citations": citations}
