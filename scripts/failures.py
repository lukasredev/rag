"""Print the questions your pipeline got wrong, from a report written by `ragkit.run --out`.

    python scripts/failures.py report.json                 # 20 failures, all types
    python scripts/failures.py report.json --type boolean  # only one question type
    python scripts/failures.py report.json --n 50 --full   # more, with whole answers and citations

Each failure shows the expected answer and where the evidence is (citations marked ✓ are right). Decide for each:
did retrieval miss the evidence, or was it found and the answer still went wrong? RAG Studio shows the full trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Question fields that hold the ids a good pipeline retrieves/cites, per track.
EVIDENCE_FIELDS = ["evidence_ids", "gold_sections", "oracle_man", "gold_id", "evidence"]  # rfc: "evidence" is a quote
ROOT = Path(__file__).resolve().parent.parent


def evidence_ids(q: dict) -> list[str]:
    for field in EVIDENCE_FIELDS:
        if q.get(field):
            ids, todo = [], [q[field]]
            while todo:  # papers has one evidence list per annotator
                x = todo.pop(0)
                todo[:0] = x if isinstance(x, list) else []
                if isinstance(x, str) and x not in ids:
                    ids.append(x)
            return ids
    return []


def matches(track: str, cite: str, evidence: str) -> bool:
    """Would the scorer count `cite` as this piece of evidence? For rfc it follows tracks/rfc/track.py:
    "rfc9110#section-4.2" and parent/child sections count; a bare id like "rfc768" is a whole document."""
    if track != "rfc":
        return cite == evidence
    doc, sec = _rfc().parse_citation(cite)
    gold_doc, _, gold_sec = evidence.partition("#")
    if not gold_sec:
        return doc == gold_doc
    return doc == gold_doc and bool(sec) and _rfc()._section_match(sec, gold_sec)


def _rfc():
    sys.path.insert(0, str(ROOT))
    from ragkit.run import load_track

    if "track_rfc" not in sys.modules:
        load_track("rfc")
    return sys.modules["track_rfc"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", type=Path)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--type", help="only this question type")
    p.add_argument("--full", action="store_true", help="whole answers and all citations")
    args = p.parse_args()

    report = json.loads(args.report.read_text())
    primary = report["primary"]
    gold = {}
    test = ROOT / "tracks" / report["track"] / "data" / "test.jsonl"
    if test.exists():
        gold = {q["id"]: q for q in map(json.loads, test.read_text().splitlines()) if q}

    wrong = [q for q in report["per_question"]
             if q["scores"].get(primary, 0) < 1 and (not args.type or q["type"] == args.type)]
    print(f"{len(wrong)} of {report['n']} questions below 1.0 on {primary}; by type: "
          + ", ".join(f"{t} {n}" for t, n in Counter(q["type"] for q in wrong).most_common()))
    cut = None if args.full else 200
    for q in wrong[: args.n]:
        pred = q["prediction"]
        print(f"\n--- {q['type']}  {q['id']}")
        print(f"Q:        {q['question']}")
        if pred.get("error"):
            print(f"ERROR:    {pred['error']}")
        print(f"answer:   {str(pred.get('answer', ''))[:cut]}")
        g = gold.get(q["id"], {})
        expected = g.get("answers") or g.get("answer") or g.get("gold_id")
        if expected:
            print(f"expected: {str(expected)[:cut]}")
        cites = [str(c) for c in pred.get("citations") or []]
        evidence = evidence_ids(g)
        is_quote = bool(evidence) and " " in evidence[0]  # rfc without sections: a quote, not an id
        targets = (g.get("gold_docs") or []) if is_quote else evidence  # what a citation has to hit
        k = 10 if report["track"] == "codesearch" else 5  # how many citations the track scores
        hit = lambda c: any(matches(report["track"], c, e) for e in targets)  # noqa: E731
        print(f"cited:    {', '.join(c + (' ✓' if hit(c) else '') for c in (cites if args.full else cites[:k]))}")
        if is_quote:
            print(f"evidence: {', '.join(targets) + ': ' if targets else ''}\"{evidence[0][:cut]}\"")
        elif evidence:
            found = sum(any(matches(report["track"], c, e) for c in cites[:k]) for e in evidence)
            print(f"evidence: {', '.join(evidence[:8])}  ({found} of {len(evidence)} among your first {k} citations)")
        print("scores:   " + "  ".join(f"{k}={v:.2f}" for k, v in q["scores"].items()))


if __name__ == "__main__":
    main()
