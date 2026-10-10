"""Starter pipeline for the Shell Command Helper track. It runs end to end -- and is deliberately weak.

What it does: BM25 over all man-page paragraphs, top 3 paragraphs go into a minimal prompt.
Look at the TODOs; every one of them is worth points on the leaderboard.

Run it:
    python -m ragkit.run --track shell --pipeline tracks/shell/skeleton/pipeline.py --limit 20
"""

import json
from pathlib import Path

from ragkit import trace  # only records anything inside RAG Studio
from ragkit.bm25 import BM25
from ragkit.llm import chat
from ragkit.text import tokenize


class Pipeline:
    def __init__(self, data_dir: Path):
        # docs.jsonl: one man-page paragraph per line: {"id": "tar_12", "cmd": "tar", "text": "..."}
        self.docs = [json.loads(line) for line in open(data_dir / "docs.jsonl")]
        # TODO: paragraphs are tiny (~18 words) and lose their context. Which command do they belong to?
        #       Would it help to index the command name together with the text, or whole commands?
        self.index = BM25([tokenize(d["text"]) for d in self.docs])
        # TODO: data/commands.jsonl (one line per command) and data/train_examples.jsonl (intent -> command
        #       pairs for OTHER commands) are unused here. How could they help?

    def answer(self, question: dict) -> dict:
        hits = self.index.search(tokenize(question["question"]), k=3)
        docs = [self.docs[i] for i, _ in hits]
        trace.log("retrieve", "BM25 man-page paragraphs", hits=[{"id": d["id"], "score": s, "text": d["text"]} for d, (_, s) in zip(docs, hits)])
        context = "\n".join(f"[{d['cmd']}] {d['text']}" for d in docs)
        # TODO: the prompt says nothing about the output format. The scorer wants ONE command line,
        #       with {{placeholders}} for values the user has to fill in (see README).
        prompt = f"Documentation:\n{context}\n\nRequest: {question['question']}\n\nWhich bash command does this?"
        answer = chat(prompt)
        # TODO: the model may answer with prose, several alternatives or a code block. Clean it up?
        return {"answer": answer, "citations": [d["id"] for d in docs]}
