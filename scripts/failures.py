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
from collections import Counter
from pathlib import Path

# Question fields that hold the ids a good pipeline retrieves/cites, per track.
EVIDENCE_FIELDS = ["evidence_ids", "gold_sections", "oracle_man", "evidence"]  # rfc: "evidence" is a quote


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
    test = Path(__file__).resolve().parent.parent / "tracks" / report["track"] / "data" / "test.jsonl"
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
        cites = pred.get("citations") or []
        evidence = evidence_ids(g)
        print(f"cited:    {', '.join(c + (' ✓' if c in evidence else '') for c in (cites if args.full else cites[:5]))}")
        if evidence and " " in evidence[0]:  # a quote, not an id
            print(f"evidence: \"{evidence[0][:cut]}\"")
        elif evidence:
            found = sum(e in cites[:5] for e in evidence)
            print(f"evidence: {', '.join(evidence[:8])}  ({found} of {len(evidence)} among your first 5 citations)")
        print("scores:   " + "  ".join(f"{k}={v:.2f}" for k, v in q["scores"].items()))


if __name__ == "__main__":
    main()
