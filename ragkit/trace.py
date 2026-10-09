"""Record what a pipeline does for one question, so RAG Studio can show it step by step.

ragkit records its own steps automatically (every chat(), query embedding and BM25 search).
Add your own steps with `trace.log`, e.g. to show which chunks you retrieved:

    from ragkit import trace
    trace.log("retrieve", "top chunks after fusion",
              hits=[{"id": doc_id, "score": score, "text": chunk} for ...])

Outside RAG Studio nothing is recorded and `log` costs nothing. Steps logged in threads that your
pipeline starts itself are not recorded (each thread has its own trace).
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager

_current: contextvars.ContextVar[list | None] = contextvars.ContextVar("ragkit_trace", default=None)
MAX_TEXT = 20000  # characters kept per text field, so one trace stays small


def active() -> bool:
    return _current.get() is not None


def _clip(value, depth=0):
    if isinstance(value, str):
        return value if len(value) <= MAX_TEXT else value[:MAX_TEXT] + f"... [{len(value) - MAX_TEXT} more characters]"
    if depth > 4:
        return repr(value)[:200]
    if isinstance(value, dict):
        return {str(k): _clip(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(v, depth + 1) for v in list(value)[:200]]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    try:  # numpy scalars and similar
        return value.item()
    except Exception:  # noqa: BLE001
        return repr(value)[:200]


def log(step: str, title: str = "", **data) -> None:
    """Add one step to the current trace. `step` names the kind of step, e.g. "retrieve", "rerank",
    "decompose", "filter"; `hits=[{"id", "score", "text"}, ...]` is shown as a ranked list."""
    events = _current.get()
    if events is None:
        return
    events.append({"step": step, "title": title, "t": round(time.time(), 3), "data": _clip(data)})


@contextmanager
def collect():
    """Record all steps logged inside the block (in this thread) into the yielded list."""
    events: list[dict] = []
    token = _current.set(events)
    try:
        yield events
    finally:
        _current.reset(token)
