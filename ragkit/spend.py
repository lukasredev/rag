"""Spend cap: count what LLM calls cost and refuse calls that would go over a budget.

    RAG_MAX_USD=5        refuse any call that could push the total above $5 (default: 5)
    RAG_MAX_USD=off      no cap

    python -m ragkit.spend            # how much has been spent (per model)
    python -m ragkit.spend --reset    # start counting from zero again

The total lives in <RAG_CACHE_DIR>/spend.sqlite, so it adds up across runs and processes on this machine.
Cached answers are free and don't count. Prices are estimates: check your provider's price list.

This only guards code that calls the API through ragkit on this machine. The real hard limit is the spend
limit of the API key in your provider's console; set that too.
"""

from __future__ import annotations

import argparse
import sqlite3
import threading
import time

from . import config

# USD per million input / output tokens.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (3.00, 15.00),
    "claude-opus-5-5": (4.00, 20.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
}
# Models we don't know a price for are charged at this rate (deliberately high), unless they run on a
# local OpenAI-compatible server (OPENAI_BASE_URL pointing at localhost), which is free.
UNKNOWN_PRICE = (5.00, 25.00)
DEFAULT_CAP = "5"
CHARS_PER_TOKEN = 3  # conservative: real text is closer to 4 characters per token

_lock = threading.Lock()
_reserved = 0.0  # worst-case cost of calls in flight in this process


class SpendCapError(RuntimeError):
    pass


def price(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]
    base = config.get("OPENAI_BASE_URL") or ""
    if not model.startswith("claude") and ("localhost" in base or "127.0.0.1" in base):
        return (0.0, 0.0)
    return UNKNOWN_PRICE


def cost(model: str, input_tokens: float, output_tokens: float) -> float:
    p_in, p_out = price(model)
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


def cap() -> float | None:
    value = (config.get("RAG_MAX_USD", DEFAULT_CAP) or "").strip().lower()
    return None if value in ("off", "none", "") else float(value)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(config.cache_dir() / "spend.sqlite", timeout=30)
    conn.execute("CREATE TABLE IF NOT EXISTS spend (t REAL, model TEXT, input_tokens INT, output_tokens INT, usd REAL)")
    return conn


def total() -> float:
    with _db() as conn:
        return conn.execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0]


def by_model() -> list[tuple[str, int, int, int, float]]:
    with _db() as conn:
        return conn.execute(
            "SELECT model, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(usd) FROM spend GROUP BY model"
        ).fetchall()


def reserve(model: str, prompt_chars: int, max_tokens: int) -> float:
    """Called before a request: raise SpendCapError if its worst case could exceed the cap.
    Returns the reserved amount, to pass to `settle` afterwards."""
    global _reserved
    limit = cap()
    worst = cost(model, prompt_chars / CHARS_PER_TOKEN, max_tokens)
    if limit is None:
        return 0.0
    with _lock:
        spent = total()
        if spent + _reserved + worst > limit:
            raise SpendCapError(
                f"Spend cap reached: ${spent:.2f} spent of RAG_MAX_USD=${limit:.2f} "
                f"(this call could cost up to ${worst:.4f}). Raise RAG_MAX_USD (or set it to 'off') to continue; "
                f"see `python -m ragkit.spend`."
            )
        _reserved += worst
    return worst


def settle(reserved: float, model: str, input_tokens: int, output_tokens: int) -> float:
    """Called after a request: record what it really cost and release the reservation."""
    global _reserved
    usd = cost(model, input_tokens, output_tokens)
    with _lock:
        _reserved = max(0.0, _reserved - reserved)
        with _db() as conn:
            conn.execute("INSERT INTO spend VALUES (?, ?, ?, ?, ?)", (time.time(), model, input_tokens, output_tokens, usd))
    return usd


def release(reserved: float) -> None:
    """Called when a request failed: nothing was spent (or we can't know), release the reservation."""
    global _reserved
    with _lock:
        _reserved = max(0.0, _reserved - reserved)


def main():
    p = argparse.ArgumentParser(description="Show or reset the LLM spend counted by ragkit")
    p.add_argument("--reset", action="store_true", help="start counting from zero")
    args = p.parse_args()
    if args.reset:
        with _db() as conn:
            conn.execute("DELETE FROM spend")
        print("spend counter reset")
    limit = cap()
    print(f"spent ${total():.4f} of {'no cap' if limit is None else f'RAG_MAX_USD=${limit:.2f}'}"
          f"  ({config.cache_dir() / 'spend.sqlite'})")
    for model, calls, n_in, n_out, usd in by_model():
        print(f"  {model:<22} {calls:>7} calls {n_in:>12,} in {n_out:>11,} out  ${usd:.4f}")


if __name__ == "__main__":
    main()
