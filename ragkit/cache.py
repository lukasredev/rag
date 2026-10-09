"""A tiny thread-safe key/value store on top of SQLite, used to cache embeddings and LLM calls.

Re-embedding the same corpus on every run wastes time and money, so every embedding
and (optionally) every LLM response is stored under a hash of its inputs.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path


def stable_hash(*parts) -> str:
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class KVCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._path = str(path)
        self._write_lock = threading.Lock()
        conn = self._conn()
        conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path, timeout=60)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def get(self, key: str) -> bytes | None:
        row = self._conn().execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return row[0] if row else None

    def get_many(self, keys: list[str]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        conn = self._conn()
        for i in range(0, len(keys), 500):
            batch = keys[i : i + 500]
            marks = ",".join("?" * len(batch))
            for k, v in conn.execute(f"SELECT k, v FROM kv WHERE k IN ({marks})", batch):
                out[k] = v
        return out

    def set_many(self, items: dict[str, bytes]) -> None:
        if not items:
            return
        with self._write_lock:
            conn = self._conn()
            conn.executemany("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", list(items.items()))
            conn.commit()

    def set(self, key: str, value: bytes) -> None:
        self.set_many({key: value})
