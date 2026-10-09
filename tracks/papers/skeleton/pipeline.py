"""Starter pipeline for "Ask the Paper" -- runs end to end, but is deliberately weak.

What it does:
  * flattens every paragraph of every paper in data/papers.jsonl into one big list
  * BM25 keyword search over ALL papers (it ignores which paper the question is about!)
  * gives the top 3 paragraphs to the LLM with a bare-bones prompt

Run it:
  python -m ragkit.run --track papers --pipeline tracks/papers/skeleton/pipeline.py --limit 20

Each TODO below is a decision worth thinking about -- measure before/after on test.jsonl.
"""

from __future__ import annotations

import json
import pathlib

from ragkit import trace  # only records anything inside RAG Studio
from ragkit.bm25 import BM25
from ragkit.llm import chat
from ragkit.text import tokenize


class Pipeline:
    def __init__(self, data_dir: pathlib.Path):
        self.chunks = []  # list of {"id", "paper_id", "text"}
        with open(data_dir / "papers.jsonl") as f:
            for line in f:
                paper = json.loads(line)
                # TODO: section names, the abstract and the figure/table captions are thrown away here.
                for section in paper["sections"]:
                    for para in section["paragraphs"]:
                        self.chunks.append({"id": para["id"], "paper_id": paper["paper_id"], "text": para["text"]})
        # TODO: one index over 200 papers -- but every question tells you which paper it is about...
        self.bm25 = BM25([tokenize(c["text"]) for c in self.chunks])

    def answer(self, question: dict) -> dict:
        # question = {"id", "question", "paper_id", "title"}
        hits = self.bm25.search(tokenize(question["question"]), k=3)  # TODO: k? dense retrieval? hybrid?
        context = [self.chunks[i] for i, _ in hits]
        trace.log("retrieve", "BM25 paragraphs", hits=[{"id": c["id"], "score": s, "text": c["text"]} for c, (_, s) in zip(context, hits)])

        context_text = "\n\n".join(f"[{c['id']}] {c['text']}" for c in context)
        # TODO: what should the model do when the paper does not contain the answer? (see README: UNANSWERABLE)
        # TODO: yes/no questions, short extractive answers ...
        prompt = f"Context:\n{context_text}\n\nQuestion: {question['question']}\nAnswer:"
        answer = chat(prompt, max_tokens=500)
        return {"answer": answer, "citations": [c["id"] for c in context]}
