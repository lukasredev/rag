"""Parse an RFC plain-text file into clean sections (used by generate_questions.py, NOT given to students).

clean_rfc(raw)      -> text without page footers ("Fielding, et al.  Standards Track  [Page 12]"),
                       page headers ("RFC 9110   HTTP Semantics   June 2022"), form feeds, and runs of blank lines
parse_sections(raw) -> [{"num": "4.2.1" | "A.3" | None, "title", "text", "start_line"}]

Section headings are detected with a regex plus a *numbering-consistency* check (a heading must continue the
section numbering: next sibling, first child, or next sibling of an ancestor), which rejects the many numbered
lists, ASCII diagrams and table-of-contents lines that look like headings.
"""

from __future__ import annotations

import re

_FOOTER = re.compile(r"^.*\[Page \d+\]\s*$")
_HEADER = re.compile(r"^RFC \d+\s{2,}.*\S\s*$")
_HEADING = re.compile(r"^( *)(Appendix\s+)?((?:\d+|[A-Z])(?:\.\d+)*)(\.?)(?:\s{1,6})([A-Za-z0-9\"'(].{0,90}?)\s*$")
_TOC_LINE = re.compile(r"(\.\s?){3,}\s*\d+\s*$|\s{2,}\d+\s*$|\.\d+$")


def clean_rfc(raw: str) -> str:
    out: list[str] = []
    after_ff = False
    for line in raw.replace("\r", "").split("\n"):
        if "\f" in line:
            after_ff = True
            line = line.replace("\f", "")
            if not line.strip():
                continue
        if _FOOTER.match(line):
            continue
        if after_ff and line.strip():
            after_ff = False
            if _HEADER.match(line):
                continue
        out.append(line.rstrip())
    text = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def _key(num: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 1000 + ord(p) for p in num.split("."))


def _follows(prev: tuple | None, cur: tuple) -> bool:
    if prev is None:
        return cur in ((1,), (1000 + ord("A"),)) or cur == (0,)
    if cur[0] >= 1000 and len(cur) == 1:  # appendix letter
        return prev[0] < 1000 or cur[0] == prev[0] + 1
    if len(cur) == len(prev) + 1 and cur[:-1] == prev and cur[-1] == 1:  # first child
        return True
    for depth in range(len(prev), 0, -1):  # next sibling of prev or of one of its ancestors
        if len(cur) == depth and cur[:-1] == prev[: depth - 1] and cur[-1] == prev[depth - 1] + 1:
            return True
    return False


_TOC_ENTRY = re.compile(r"^\s*(Appendix\s+)?((?:\d+|[A-Z])(?:\.\d+)*)(\.?)\s+(\S.*?)(?:[\s.]+(?:\d+|[ivx]+))?\s*$")


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower())[:12]


def _toc(lines: list[str]) -> tuple[dict[str, tuple[int, str]], int]:
    """Section numbers listed in the table of contents -> (order, title), and the line where the TOC ends.

    The TOC starts at a "Table of Contents" line and ends where its first entry (usually "1. Introduction")
    appears again as a real heading."""
    start = next((i for i, l in enumerate(lines[:400]) if l.strip().lower() == "table of contents"), None)
    if start is None:
        return {}, 0
    toc: dict[str, tuple[int, str]] = {}
    first = None
    for i in range(start + 1, len(lines)):
        line = lines[i]
        m = _TOC_ENTRY.match(line)
        if not m:
            continue
        appendix, num, dot, title = m.groups()
        if num[0].isalpha() and not (appendix or "." in num or dot):
            continue
        if first is None:
            first = (num, _norm_title(title))
        elif (num, _norm_title(title)) == first:
            return (toc, i - 1) if len(toc) >= 4 else ({}, 0)
        toc.setdefault(num, (len(toc), title))
    return {}, 0


def parse_sections(raw: str) -> list[dict]:
    lines = clean_rfc(raw).split("\n")
    toc, toc_end = _toc(lines)
    heads: list[tuple[int, str, str]] = []
    prev = None
    last_order = -1
    for i, line in enumerate(lines):
        if i <= toc_end and toc:
            continue
        m = _HEADING.match(line)
        if not m or (i > 0 and lines[i - 1].strip()):
            continue
        indent, appendix, num, dot, title = m.groups()
        if _TOC_LINE.search(line) or len(title) > 80 or len(indent) > 40:
            continue
        if num[0].isalpha() and not (appendix or "." in num or dot):
            continue  # "A full implementation ..." is a sentence, not Appendix A
        if toc:  # the table of contents is the ground truth for which headings exist
            entry = toc.get(num)
            if not entry or entry[0] <= last_order or _norm_title(entry[1])[:8] != _norm_title(title)[:8]:
                continue
            last_order = entry[0]
            heads.append((i, num, title.strip()))
            continue
        if title[0].isdigit() or (len(indent) > 3 and not (title.isupper() or appendix)):
            continue  # deeply indented: only centred ALL-CAPS headings (RFC 791 style)
        key = _key(num)
        if _follows(prev, key):
            heads.append((i, num, title.strip()))
            prev = key
    sections = []
    first = heads[0][0] if heads else len(lines)
    front = "\n".join(lines[:first]).strip()
    if front:
        sections.append({"num": None, "title": "Front matter", "text": front, "start_line": 0})
    for j, (i, num, title) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
        body = "\n".join(lines[i + 1 : end]).strip()
        sections.append({"num": num, "title": title, "text": body, "start_line": i})
    return sections


if __name__ == "__main__":
    import sys
    from pathlib import Path

    for path in sys.argv[1:]:
        secs = parse_sections(Path(path).read_text(errors="replace"))
        print(f"{path}: {len(secs)} sections; " + " | ".join(f"{s['num']} {s['title'][:25]}" for s in secs[:12]))
