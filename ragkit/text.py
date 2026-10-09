"""Text helpers: tokenising and chunking. Deliberately simple, so you can read and change them."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an and are as at be by for from has have how i in is it its of on or that the this to
    was were what when where which who why will with do does did can you your their they them""".split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN.findall(text.lower())
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def chunk_chars(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Fixed-size character windows. The simplest possible chunker (and often a bad one)."""
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]


def chunk_words(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Fixed-size windows of words (does not cut words in half)."""
    words = text.split()
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    chunks = []
    for start in range(0, max(len(words), 1), size - overlap):
        piece = words[start : start + size]
        if piece:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def split_paragraphs(text: str, min_chars: int = 1) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]
    return [p for p in paragraphs if len(p) >= min_chars]
