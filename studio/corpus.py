"""Corpus explorer: one browsable view of each track's documents, whatever their file format.

Every track stores its corpus differently (papers.jsonl with sections, raw RFC .txt files, man-page paragraphs,
...). An adapter turns it into documents of one shape:

    {"id", "title", "meta": {field: value}, "parts": [{"id": passage id | None, "heading": str | None, "text"}]}

and `resolve(citation_id)` maps an id a pipeline cites (a paragraph, a section, a man-page paragraph) to the
document it belongs to and the passage inside it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


class Corpus:
    unit = "document"
    facets: list[str] = []
    note = ""

    def __init__(self, data_dir: Path):
        self.docs: list[dict] = self.load(data_dir)
        self.by_id = {d["id"]: d for d in self.docs}
        for d in self.docs:
            d["_text"] = (d["title"] + "\n" + "\n".join(p["text"] for p in d["parts"])).lower()
            d["words"] = sum(len(p["text"].split()) for p in d["parts"])

    def load(self, data_dir: Path) -> list[dict]:
        raise NotImplementedError

    def resolve(self, cid: str) -> tuple[str | None, str | None]:
        return (cid, None) if cid in self.by_id else (None, None)

    def evidence(self, q: dict) -> list[dict]:
        """The documents a test question needs and, where known, the passages inside them:
        [{"doc", "role": "gold" | ..., "passages": [{"anchor", "label", "text", "quote"}]}]"""
        return []

    def part_text(self, doc: str, part_id: str) -> str:
        return next((p["text"] for p in self.by_id[doc]["parts"] if p["id"] == part_id), "")


class Papers(Corpus):
    unit = "paper"
    note = "Each question names its paper (paper_id); citations are paragraph ids paper:section:paragraph."

    def load(self, data_dir):
        docs = []
        for line in open(data_dir / "papers.jsonl"):
            p = json.loads(line)
            parts = [{"id": None, "heading": "Abstract", "text": p.get("abstract") or ""}]
            for s in p["sections"]:
                for i, para in enumerate(s["paragraphs"]):
                    parts.append({"id": para["id"], "heading": s.get("name") if i == 0 else None, "text": para["text"]})
            for i, cap in enumerate(p.get("captions") or []):
                text = cap.get("text") if isinstance(cap, dict) else str(cap)
                cid = cap.get("id") if isinstance(cap, dict) else f"{p['paper_id']}:cap:{i}"
                parts.append({"id": cid, "heading": "Figure / table captions" if i == 0 else None, "text": text})
            docs.append({"id": p["paper_id"], "title": p["title"],
                         "meta": {"sections": len(p["sections"]), "paragraphs": len(parts) - 1}, "parts": parts})
        return docs

    def resolve(self, cid):
        doc = cid.split(":", 1)[0]
        return (doc, cid if ":" in cid else None) if doc in self.by_id else (None, None)

    def evidence(self, q):
        doc = q.get("paper_id")
        if doc not in self.by_id:
            return []
        ids = list(dict.fromkeys(i for annot in q.get("evidence") or [] for i in (annot or [])))
        return [{"doc": doc, "role": "gold" if ids else "asked about (no evidence: unanswerable)",
                 "passages": [{"anchor": i, "label": i, "text": self.part_text(doc, i)} for i in ids]}]


class TechNews(Corpus):
    unit = "article"
    facets = ["source", "category", "month"]

    def load(self, data_dir):
        docs = []
        for line in open(data_dir / "articles.jsonl"):
            a = json.loads(line)
            meta = {"source": a.get("source"), "category": a.get("category"),
                    "published": (a.get("published_at") or "")[:10], "month": (a.get("published_at") or "")[:7],
                    "author": a.get("author"), "url": a.get("url")}
            docs.append({"id": a["id"], "title": a["title"], "meta": meta,
                         "parts": [{"id": None, "heading": None, "text": a["body"]}]})
        return docs


class TechNewsEvidence:
    def evidence(self, q):
        out = []
        for doc in q.get("evidence_ids") or []:
            if doc in self.by_id:
                facts = [f["fact"] for f in q.get("evidence_facts") or [] if f.get("doc_id") == doc]
                out.append({"doc": doc, "role": "gold",
                            "passages": [{"anchor": None, "label": "fact", "text": f, "quote": True} for f in facts]})
        return out


TechNews.evidence = TechNewsEvidence.evidence


def _squash(text: str) -> str:
    return " ".join(text.split()).lower()


class RFC(Corpus):
    unit = "RFC"
    facets = ["status", "decade"]
    note = "Raw text exactly as the pipeline gets it (page headers, footers and all). Citations: rfc9110 or rfc9110#4.2."

    def load(self, data_dir):
        docs = []
        for d in json.loads((data_dir / "index.json").read_text()):
            raw = (data_dir / "rfcs" / f"{d['id']}.txt").read_text(errors="replace")
            meta = {"number": d["number"], "date": d.get("date"), "status": d.get("status"),
                    "decade": f"{d['year'] // 10 * 10}s" if d.get("year") else None}
            for k in ("obsoletes", "obsoleted_by", "updates", "updated_by"):
                if d.get(k):
                    meta[k.replace("_", " ")] = ", ".join(d[k])
            docs.append({"id": d["id"], "title": d["title"], "meta": meta,
                         "parts": [{"id": None, "heading": None, "text": raw}]})
        return docs

    def resolve(self, cid):
        doc, _, section = cid.partition("#")
        return (doc, section or None) if doc in self.by_id else (None, None)

    def evidence(self, q):
        gold = [d for d in q.get("gold_docs") or [] if d in self.by_id]
        sections: dict[str, list] = {}
        for src in q.get("source_sections") or []:  # "rfc9113 §3.3 Starting HTTP/2 with Prior Knowledge"
            m = re.match(r"(rfc\d+)\s+§(\S+)\s*(.*)", src)
            if m and m.group(1) in self.by_id:
                sec = None if m.group(2) == "-" else m.group(2)
                sections.setdefault(m.group(1), []).append(
                    {"anchor": sec, "label": f"§{m.group(2)} {m.group(3)}".strip(), "text": "", "section": True})
        docs = list(dict.fromkeys(gold + list(sections)))
        quotes: dict[str, list] = {d: [] for d in docs}
        for quote in q.get("evidence") or []:
            home = next((d for d in docs if _squash(quote) in self.by_id[d]["_text"].replace("\n", " ")
                         or _squash(quote) in _squash(self.by_id[d]["parts"][0]["text"])), gold[0] if gold else None)
            if home:
                quotes.setdefault(home, []).append({"anchor": None, "label": "quote", "text": quote, "quote": True})
        return [{"doc": d, "role": "gold" if d in gold else "older / other version (not the answer source)",
                 "passages": sections.get(d, []) + quotes.get(d, [])} for d in docs]


class Shell(Corpus):
    unit = "command"
    facets = ["has tldr examples"]
    note = "One document per command: its man-page paragraphs (each paragraph is a separate retrievable unit)."

    def load(self, data_dir):
        summaries = {}
        if (data_dir / "commands.jsonl").exists():
            for line in open(data_dir / "commands.jsonl"):
                c = json.loads(line)
                summaries[c["cmd"]] = c.get("summary") or ""
        examples: dict[str, list] = {}
        if (data_dir / "train_examples.jsonl").exists():
            for line in open(data_dir / "train_examples.jsonl"):
                e = json.loads(line)
                examples.setdefault(e.get("cmd_name"), []).append(e)
        grouped: dict[str, list] = {}
        self.para_to_cmd = {}
        for line in open(data_dir / "docs.jsonl"):
            p = json.loads(line)
            grouped.setdefault(p["cmd"], []).append({"id": p["id"], "heading": None, "text": p["text"]})
            self.para_to_cmd[p["id"]] = p["cmd"]
        docs = []
        for cmd, parts in grouped.items():
            ex = examples.get(cmd, [])
            if ex:
                parts = parts + [{"id": None, "heading": "tldr examples (data/train_examples.jsonl)" if i == 0 else None,
                                  "text": f"{e['nl']}\n    {e['cmd']}"} for i, e in enumerate(ex)]
            docs.append({"id": cmd, "title": f"{cmd}: {summaries.get(cmd, '')}".rstrip(": "),
                         "meta": {"paragraphs": len(grouped[cmd]), "has tldr examples": "yes" if ex else "no"},
                         "parts": parts})
        return docs

    def resolve(self, cid):
        if cid in self.by_id:
            return cid, None
        cmd = getattr(self, "para_to_cmd", {}).get(cid)
        return (cmd, cid) if cmd else (None, None)

    def evidence(self, q):
        cmd = q.get("cmd_name")
        if cmd not in self.by_id:
            return []
        paras = [{"anchor": pid, "label": pid, "text": self.part_text(cmd, pid)} for pid in q.get("oracle_man") or []]
        return [{"doc": cmd, "role": "gold", "passages": paras}]


class CodeSearch(Corpus):
    unit = "function"
    facets = ["repo"]
    note = "Docstrings were stripped; the questions are (rewritten) docstrings."

    def load(self, data_dir):
        docs = []
        for line in open(data_dir / "corpus.jsonl"):
            f = json.loads(line)
            docs.append({"id": f["id"], "title": f["func_name"],
                         "meta": {"repo": f.get("repo"), "path": f.get("path"), "url": f.get("url")},
                         "parts": [{"id": None, "heading": None, "text": f["code"], "code": True}]})
        return docs

    def evidence(self, q):
        doc = q.get("gold_id")
        return [{"doc": doc, "role": "gold", "passages": []}] if doc in self.by_id else []


ADAPTERS = {"papers": Papers, "technews": TechNews, "rfc": RFC, "shell": Shell, "codesearch": CodeSearch}


def length_histogram(words: list[int]) -> list[dict]:
    """Bucket document lengths on a log scale (10, 20, 50, 100, 200, 500, ...)."""
    edges = [0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 10**9]
    counts = Counter()
    for w in words:
        for lo, hi in zip(edges, edges[1:]):
            if lo <= w < hi:
                counts[(lo, hi)] += 1
                break
    used = [b for b in zip(edges, edges[1:]) if counts[b]]
    if not used:
        return []
    first, last = edges.index(used[0][0]), edges.index(used[-1][0])
    return [{"lo": edges[i], "hi": edges[i + 1], "n": counts[(edges[i], edges[i + 1])]} for i in range(first, last + 1)]


def search(corpus: Corpus, q: str, filters: dict[str, str]) -> list[tuple[float, dict]]:
    terms = [t for t in re.findall(r"\w+", q.lower()) if t]
    patterns = [re.compile(r"\b" + re.escape(t) + r"\b") for t in terms]  # whole words: "error" is not "terror"
    out = []
    for d in corpus.docs:
        if any(str(d["meta"].get(k)) != v for k, v in filters.items()):
            continue
        if terms:
            counts = [len(rx.findall(d["_text"])) for rx in patterns]
            if not all(counts):
                continue
            title = d["title"].lower()
            score = sum(counts) + 20 * sum(t in title for t in terms)
        else:
            score = 0
        out.append((score, d))
    if terms:
        out.sort(key=lambda x: -x[0])
    return out


def snippet(d: dict, q: str, width: int = 220) -> str:
    text = " ".join(" ".join(p["text"] for p in d["parts"]).split())
    terms = [t for t in re.findall(r"\w+", q.lower()) if t]
    pos = min((i for i in (text.lower().find(t) for t in terms) if i >= 0), default=0)
    start = max(0, pos - width // 3)
    return ("…" if start else "") + text[start:start + width] + ("…" if start + width < len(text) else "")
