"""Track `papers`: Ask the Paper (Qasper). Self-contained: imports only ragkit + stdlib.

Metrics (all in [0, 1], averaged over questions; see README.md):
    correct            PRIMARY. unanswerable -> answer is the abstention token; boolean -> yes/no matches;
                       extractive/abstractive -> max token-F1 >= 0.8 against a reference, else LLM judge.
    answer_f1          Qasper-style max token F1 over all references (yes/no and abstention handled).
    evidence_recall@5  paragraph-level recall of the first 5 citations, best over annotators
                       (1.0 for unanswerable questions, which have no evidence -- read it per type).
    abstained          1 if the answer is the abstention token. Per type: on `unanswerable` this is good,
                       on every other type it is a false abstention.
"""

from __future__ import annotations

import re

from ragkit.metrics import best_of, llm_judge, normalize_answer, recall_at_k, token_f1

NAME = "papers"
TITLE = "Ask the Paper (Qasper)"
PRIMARY = "correct"
PUBLIC_FIELDS = ["id", "question", "paper_id", "title"]
ABSTAIN = "UNANSWERABLE"
MAX_CITATIONS = 5
F1_SHORTCUT = 0.8


def is_abstention(answer: str) -> bool:
    """Exactly the token, case-insensitive; surrounding whitespace/quotes and a trailing period are ignored."""
    a = (answer or "").strip().strip("\"'`*").strip().rstrip(".").strip()
    return a.upper() == ABSTAIN


def yes_no(answer: str) -> str | None:
    """'yes' / 'no' if the answer starts with that word, else None."""
    m = re.match(r"\W*(yes|no)\b", (answer or "").lower())
    return m.group(1) if m else None


def evidence_recall(citations: list[str], evidence: list[list[str]], k: int = MAX_CITATIONS) -> float:
    if not evidence:
        return 1.0
    cited = list(dict.fromkeys(citations))[:k]
    return max(recall_at_k(cited, set(ev), k) for ev in evidence)


def score(question: dict, prediction: dict) -> dict[str, float]:
    answer = str(prediction.get("answer") or "").strip()
    citations = [str(c) for c in (prediction.get("citations") or [])][:MAX_CITATIONS]
    qtype = question["type"]
    refs = question["answers"]
    abstained = is_abstention(answer)

    if qtype == "unanswerable":
        correct = f1 = float(abstained)
    elif qtype == "boolean":
        correct = f1 = float(not abstained and yes_no(answer) == question["yes_no"])
    else:
        f1 = 0.0 if abstained or not answer else best_of(token_f1, answer, refs)
        if abstained or not answer:
            correct = 0.0
        elif f1 >= F1_SHORTCUT:
            correct = 1.0
        else:
            correct = llm_judge(question["question"], refs, answer)

    return {
        "correct": float(correct),
        "answer_f1": float(f1),
        "evidence_recall@5": evidence_recall(citations, question.get("evidence") or []),
        "abstained": float(abstained),
    }


if __name__ == "__main__":  # tiny self-test of the deterministic parts
    assert is_abstention(" unanswerable. ") and not is_abstention("UNANSWERABLE because ...")
    assert yes_no("Yes, they do.") == "yes" and yes_no("No.") == "no" and yes_no("Not really") is None
    q = {"type": "boolean", "answers": ["No"], "yes_no": "no", "evidence": [["p:0:1"]], "question": "?"}
    assert score(q, {"answer": "no", "citations": ["p:0:1"]})["correct"] == 1.0
    assert normalize_answer("The BERT model.") == "bert model"
    print("ok")
