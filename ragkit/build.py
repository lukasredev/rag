"""Which version of the code is running, for the footer of Studio and the evaluation server.

The Docker images get the commit at build time (RAG_BUILD_SHA, and RAG_BUILD_REPO for a public repo whose
commits can be linked); in a git checkout it is read from git; otherwise it is "dev".
"""

from __future__ import annotations

import functools
import html
import os
import subprocess
from pathlib import Path

AUTHOR, AUTHOR_URL = "lukasredev", "https://github.com/lukasredev"


@functools.lru_cache(maxsize=1)
def build_info() -> dict:
    sha = os.environ.get("RAG_BUILD_SHA", "")
    if not sha or sha == "dev":
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
                                 capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            sha = ""
    repo = os.environ.get("RAG_BUILD_REPO", "").rstrip("/")
    return {"sha": sha[:7] or "dev", "url": f"{repo}/commit/{sha}" if repo and sha else None,
            "author": AUTHOR, "author_url": AUTHOR_URL}


def footer_html() -> str:
    b = build_info()
    commit = f"<code>{html.escape(b['sha'])}</code>"
    if b["url"]:
        commit = f'<a href="{html.escape(b["url"])}">{commit}</a>'
    return (f'built with ❤️ in 🇨🇭 by <a href="{b["author_url"]}">{b["author"]}</a> · commit {commit}')
