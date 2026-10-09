"""Track `technews`: Tech News Analyst on MultiHop-RAG (609 news articles, Sept-Dec 2023).

Scoring (see README):
  correct            PRIMARY. null_query -> 1 iff the answer abstains ("Insufficient information").
                     Yes/No gold -> first word of the answer must be the right Yes/No.
                     Entity gold -> normalized exact match, or gold contained in a short (<= 12 tokens) answer,
                     otherwise an LLM judge (claude-haiku) on the first 30 words. Abstaining on an answerable
                     question is always wrong.
  exact              normalized exact match against the gold string (abstain/Yes/No questions: same as correct).
  abstained          1 if the answer is an abstention (useful per type: should be 1 on null, 0 elsewhere).
  evidence_recall@5  fraction of gold evidence articles among the first 5 distinct cited ids.
                     null_query (no evidence): 1 if nothing is cited, else 0.
  evidence_all@5     1 if ALL gold evidence articles are among the first 5 cited ids (same rule for null).
"""

from __future__ import annotations

from ragkit.metrics import llm_judge, normalize_answer

NAME = "technews"
TITLE = "Tech News Analyst (MultiHop-RAG)"
PRIMARY = "correct"
PUBLIC_FIELDS = ["id", "question"]

ABSTAIN = "Insufficient information"
MAX_CITATIONS = 5
SHORT_ANSWER_TOKENS = 12
JUDGE_WORDS = 30


def is_abstention(answer: str) -> bool:
    norm = normalize_answer(answer)
    return norm.startswith("insufficient information") or norm.startswith("not enough information")


def _first_word(answer: str) -> str:
    words = normalize_answer(answer).split()
    return words[0] if words else ""


def _dedupe(ids: list[str]) -> list[str]:
    out = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out


def score(question: dict, prediction: dict) -> dict[str, float]:
    answer = str(prediction.get("answer") or "")
    cited = _dedupe([str(c) for c in (prediction.get("citations") or [])])[:MAX_CITATIONS]
    gold, kind = question["answer"], question["answer_kind"]
    abstained = is_abstention(answer)

    if kind == "abstain":
        correct = exact = float(abstained)
    elif abstained:
        correct = exact = 0.0
    elif kind == "yesno":
        correct = exact = float(_first_word(answer) == gold.lower())
    else:
        norm_pred, norm_gold = normalize_answer(answer), normalize_answer(gold)
        exact = float(norm_pred == norm_gold)
        if exact:
            correct = 1.0
        elif norm_gold and len(norm_pred.split()) <= SHORT_ANSWER_TOKENS and f" {norm_gold} " in f" {norm_pred} ":
            correct = 1.0
        else:
            short = " ".join(answer.split()[:JUDGE_WORDS])
            correct = llm_judge(question["question"], [gold], short)

    relevant = set(question.get("evidence_ids") or [])
    if relevant:
        hit = len(set(cited) & relevant)
        recall, all_found = hit / len(relevant), float(hit == len(relevant))
    else:
        recall = all_found = float(not cited)

    return {
        "correct": float(correct),
        "exact": float(exact),
        "abstained": float(abstained),
        "evidence_recall@5": recall,
        "evidence_all@5": all_found,
    }
