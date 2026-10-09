"""Starter pipeline for the `rfc` track. It runs end to end -- and it is deliberately weak.

What it does: reads the RAW RFC text files, cuts them into fixed 1000-character windows (page headers,
footers and form feeds included), finds the 3 best windows with BM25, and asks the LLM to answer.

Run it:
    RAG_ENV_FILE=... python -m ragkit.run --track rfc --pipeline tracks/rfc/skeleton/pipeline.py --limit 20

Every TODO below is a place where a better decision makes a measurable difference.
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
        # data/index.json: number, title, year, obsoletes / obsoleted_by / updates ... for every RFC
        self.index = {d["id"]: d for d in json.loads((data_dir / "index.json").read_text())}

        self.chunks: list[dict] = []
        for path in sorted((data_dir / "rfcs").glob("rfc*.txt")):
            raw = path.read_text(errors="replace")
            # TODO: the raw text has page headers/footers ("Fielding, et al.  Standards Track  [Page 12]"),
            #       form feeds and a table of contents. Clean it before chunking?
            # TODO: RFCs have numbered sections ("4.2.1.  Title"). Chunking by section keeps a rule together
            #       with its exceptions -- and lets you cite "rfc9110#4.2.1".
            for text in chunk_chars(raw, size=1000, overlap=0):
                self.chunks.append({"doc": path.stem, "text": text})

        # TODO: BM25 is great for exact terms (header names, error codes, constants). Dense embeddings
        #       (ragkit.embed) are better at paraphrases. Maybe use both?
        self.bm25 = BM25([tokenize(c["text"]) for c in self.chunks])

    def answer(self, question: dict) -> dict:
        hits = self.bm25.search(tokenize(question["question"]), k=3)  # TODO: is k=3 enough?
        trace.log("retrieve", "BM25 chunks", hits=[{"id": self.chunks[i]["doc"], "score": s, "text": self.chunks[i]["text"]} for i, s in hits])
        context = "\n\n---\n\n".join(f"[{self.chunks[i]['doc']}]\n{self.chunks[i]['text']}" for i, _ in hits)

        # TODO: several RFCs are OBSOLETE (e.g. RFC 2616 -> RFC 9110/9112). Which version should win when
        #       they disagree? data/index.json knows.
        # TODO: some questions cannot be answered from the corpus -- the expected answer is then exactly
        #       "UNANSWERABLE". Tell the model when (and when not) to say that.
        prompt = f"Context:\n{context}\n\nQuestion: {question['question']}\nAnswer briefly."
        answer = chat(prompt)

        # TODO: cite what you actually used (doc ids like "rfc9110", or sections like "rfc9110#4.2").
        #       Only the first 5 citations count.
        citations = list(dict.fromkeys(self.chunks[i]["doc"] for i, _ in hits))
        return {"answer": answer, "citations": citations}
