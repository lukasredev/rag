"""Track `codesearch`: "Where is the code that ...?" over 10,000 Python functions (CodeSearchNet).

A prediction is {"answer": "<function id>", "citations": ["csn-...", ...]} -- a RANKED list of
function ids, best first. Only the first 10 citations are scored. If `citations` is empty the
`answer` id is used as a one-element ranking.

Metrics (mean over questions):
  mrr@10  1/rank of the gold function within the first 10 citations, 0 if it is not there  (PRIMARY)
  hit@1   1 if the first citation is the gold function
  hit@10  1 if the gold function is anywhere in the first 10 citations
"""

NAME = "codesearch"
TITLE = "Where is the code that...? (CodeSearchNet Python)"
PRIMARY = "mrr@10"
PUBLIC_FIELDS = ["id", "question"]
MAX_CITATIONS = 10


def ranking(prediction: dict) -> list[str]:
    cites = [str(c).strip() for c in (prediction.get("citations") or []) if str(c).strip()]
    if not cites and str(prediction.get("answer") or "").strip():
        cites = [str(prediction["answer"]).strip()]
    seen, out = set(), []
    for c in cites:  # duplicates do not get a second chance
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out[:MAX_CITATIONS]


def score(question: dict, prediction: dict) -> dict[str, float]:
    ranked = ranking(prediction)
    gold = question["gold_id"]
    rank = ranked.index(gold) + 1 if gold in ranked else None
    return {
        "mrr@10": 1.0 / rank if rank else 0.0,
        "hit@1": float(rank == 1),
        "hit@10": float(rank is not None),
    }
