"""Track `rfc`: answer questions about ~44 IETF RFCs (HTTP, TLS, TCP/IP, DNS, mail, web formats ...).

Self-contained: imports only ragkit and the stdlib (the server imports this file standalone).
"""

from __future__ import annotations

import re

from ragkit.metrics import exact_match, llm_judge

NAME = "rfc"
TITLE = "Ask the RFCs (IETF standards, incl. obsolete versions)"
PRIMARY = "correct"
PUBLIC_FIELDS = ["id", "question"]

ABSTAIN = "UNANSWERABLE"
MAX_CITATIONS = 5

# "#", "§", "section", "sec." or "appendix", optionally followed by a word like the "section-" of rfc-editor.org
# anchors ("rfc9110#section-4.2"), then the section number itself
_CITE = re.compile(
    r"rfc[\s\-_]*0*(\d+)\s*(?:(?:#|§|,?\s*(?:section\b|sec\b\.?|appendix\b))\s*"
    r"(?:(?:section|sec|appendix)\b[\s\-.]*)?([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*))?",
    re.I,
)


def parse_citation(cite: str) -> tuple[str | None, str | None]:
    """'rfc9110#4.2' / 'RFC 9110 §4.2' / 'rfc9110#section-4.2' / 'rfc9110' -> ('rfc9110', '4.2' | None)."""
    m = _CITE.search(str(cite))
    if not m:
        return None, None
    sec = m.group(2).rstrip(".") if m.group(2) else None
    return f"rfc{int(m.group(1))}", sec


def _section_match(cited: str, gold: str) -> bool:
    """Same section, or one is an ancestor of the other (citing 4.2 for gold 4.2.1 counts, and vice versa)."""
    cited, gold = cited.lower(), gold.lower()
    return cited == gold or gold.startswith(cited + ".") or cited.startswith(gold + ".")


def is_abstention(answer: str) -> bool:
    return ABSTAIN in (answer or "").upper()


def score(question: dict, prediction: dict) -> dict[str, float]:
    answer = str(prediction.get("answer") or "")
    cites = [parse_citation(c) for c in list(prediction.get("citations") or [])[:MAX_CITATIONS]]
    cited_docs = list(dict.fromkeys(d for d, _ in cites if d))
    cited_secs = [(d, s) for d, s in cites if d and s]
    gold_docs = question.get("gold_docs") or []
    gold_secs = [g.split("#", 1) for g in question.get("gold_sections") or []]
    abstained = is_abstention(answer)

    if question["type"] == "unanswerable":
        correct = float(abstained)
    elif abstained or not answer.strip():
        correct = 0.0
    elif any(exact_match(answer, g) for g in question["answers"]):
        correct = 1.0
    else:
        correct = llm_judge(question["question"], question["answers"], answer)

    if gold_docs:
        recall = len(set(cited_docs) & set(gold_docs)) / len(set(gold_docs))
    else:  # unanswerable: nothing to find (ragkit convention: recall over an empty set is 1)
        recall = 1.0
    if gold_secs:
        sec_hit = float(any(d == gd and _section_match(s, gs) for d, s in cited_secs for gd, gs in gold_secs))
    elif gold_docs:  # gold passage has no section number (RFC 768/792/826/854): fall back to the document
        sec_hit = float(bool(set(cited_docs) & set(gold_docs)))
    else:
        sec_hit = 1.0
    return {
        "correct": correct,
        "citation_recall@5": recall,
        "section_hit@5": sec_hit,
        "abstained": float(abstained),
    }
