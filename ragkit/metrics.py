"""Metrics shared by all tracks. The server uses exactly these functions, so you can score locally.

Answer metrics:    normalize_answer, exact_match, token_f1, best_of
Retrieval metrics: recall_at_k, precision_at_k, reciprocal_rank
LLM judge:         llm_judge  (for free-text answers where string matching is too strict)
"""

from __future__ import annotations

import re
import string
from collections import Counter

from . import config

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NO_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation, articles and extra whitespace."""
    text = (text or "").lower()
    text = text.translate(_NO_PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred_tokens), overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_of(metric, prediction: str, golds: list[str]) -> float:
    """Score against several acceptable gold answers and keep the best."""
    return max((metric(prediction, g) for g in golds), default=0.0)


def recall_at_k(retrieved: list[str], relevant: list[str] | set[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 1.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: list[str] | set[str], k: int) -> float:
    top, relevant = retrieved[:k], set(relevant)
    if not top:
        return 0.0
    return len([r for r in top if r in relevant]) / len(top)


def reciprocal_rank(retrieved: list[str], relevant: list[str] | set[str]) -> float:
    relevant = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


JUDGE_SYSTEM = """You grade answers of a question-answering system against reference answers.
Decide whether the predicted answer is correct: it must agree in substance with at least one
reference answer. Extra detail is fine if it does not contradict the reference. Paraphrases,
synonyms and different formatting are fine. Missing the key fact, hedging between options, or
contradicting the reference is incorrect.
Reply with JSON only: {"correct": true} or {"correct": false}."""


JUDGE_MAX_CHARS = 3000  # longer answers are cut before judging (answers are meant to be short)


def llm_judge(question: str, references: list[str], prediction: str, model: str | None = None) -> float:
    """Return 1.0 if an LLM judge thinks `prediction` matches one of the `references`, else 0.0."""
    from .llm import chat_json

    if not (prediction or "").strip():
        return 0.0
    model = model or config.get("RAG_JUDGE_MODEL", "claude-haiku-5-5")
    refs = "\n".join(f"- {r}" for r in references)
    prediction = prediction[:JUDGE_MAX_CHARS]  # the judge's cost must not grow with the answer
    prompt = f"Question: {question}\n\nReference answer(s):\n{refs}\n\nPredicted answer: {prediction}"
    verdict = chat_json(prompt, system=JUDGE_SYSTEM, model=model, max_tokens=2000, cache=True)
    return float(bool(isinstance(verdict, dict) and verdict.get("correct") is True))
