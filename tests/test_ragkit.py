"""Unit tests for the deterministic parts of ragkit (no network, no models)."""

import numpy as np
import pytest

from ragkit.bm25 import BM25
from ragkit.cache import KVCache, stable_hash
from ragkit.llm import parse_json
from ragkit.metrics import best_of, exact_match, normalize_answer, precision_at_k, recall_at_k, reciprocal_rank, token_f1
from ragkit.text import chunk_chars, chunk_words, split_paragraphs, tokenize


def test_normalize_and_exact_match():
    assert normalize_answer("The  Quick, brown fox!") == "quick brown fox"
    assert exact_match("An apple.", "apple") == 1.0
    assert exact_match("apples", "apple") == 0.0


def test_token_f1():
    assert token_f1("port 4711", "4711") == pytest.approx(2 / 3)
    assert token_f1("", "") == 1.0
    assert token_f1("x", "") == 0.0
    assert best_of(token_f1, "6 KB", ["6 kilobytes", "6 KB"]) == 1.0


def test_retrieval_metrics():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["b", "z"], 2) == 0.5
    assert recall_at_k(retrieved, [], 2) == 1.0
    assert precision_at_k(retrieved, ["a", "c"], 2) == 0.5
    assert reciprocal_rank(retrieved, ["c"]) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, ["z"]) == 0.0


def test_bm25_ranks_matching_doc_first():
    docs = ["the cat sat on the mat", "dogs chase cats", "quantum chromodynamics lecture notes"]
    index = BM25([tokenize(d) for d in docs])
    hits = index.search(tokenize("quantum lecture"), k=3)
    assert hits[0][0] == 2
    assert all(score > 0 for _, score in hits)
    assert index.search(tokenize("zzz"), k=3) == []


def test_chunkers():
    text = "x" * 2500
    chunks = chunk_chars(text, size=1000, overlap=200)
    assert len(chunks) == 4 and all(len(c) <= 1000 for c in chunks)
    words = " ".join(str(i) for i in range(450))
    wchunks = chunk_words(words, size=200, overlap=50)
    assert wchunks[0].split()[0] == "0" and wchunks[1].split()[0] == "150"
    assert wchunks[-1].split()[-1] == "449"
    assert split_paragraphs("a\n\n\nb\n \nc") == ["a", "b", "c"]
    with pytest.raises(ValueError):
        chunk_chars("abc", size=10, overlap=10)


def test_parse_json():
    assert parse_json('```json\n{"correct": true}\n```') == {"correct": True}
    assert parse_json('Sure! {"a": [1, 2]} hope that helps') == {"a": [1, 2]}
    assert parse_json("[1, 2, 3]") == [1, 2, 3]
    assert parse_json("no json here") is None
    assert parse_json('{"a": 1}\n{"b": 2}') == {"a": 1}
    assert parse_json('set {x} then {"ok": false}') == {"ok": False}


def test_kv_cache_roundtrip(tmp_path):
    cache = KVCache(tmp_path / "c.sqlite")
    vec = np.arange(4, dtype=np.float32)
    cache.set_many({"k1": vec.tobytes(), "k2": b"hello"})
    assert np.array_equal(np.frombuffer(cache.get("k1"), dtype=np.float32), vec)
    assert cache.get_many(["k2", "missing"]) == {"k2": b"hello"}
    assert stable_hash("a", {"x": 1}) == stable_hash("a", {"x": 1})
    assert stable_hash("a") != stable_hash("b")


def test_trace_records_only_inside_collect():
    from ragkit import trace

    index = BM25([["red", "apple"], ["green", "pear"]])
    trace.log("ignored")  # outside collect(): no-op
    with trace.collect() as events:
        hits = index.search(["apple", "unknownterm"], k=2)
        trace.log("retrieve", "custom", hits=[{"id": "d0", "score": hits[0][1], "text": "x" * 30000}])
    assert [e["step"] for e in events] == ["bm25", "retrieve"]
    assert events[0]["data"]["terms"]["unknownterm"] is None
    assert len(events[1]["data"]["hits"][0]["text"]) < 21000
    assert not trace.active()


def test_spend_cap_refuses_calls_over_budget(tmp_path, monkeypatch):
    from ragkit import spend

    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_MAX_USD", "0.01")
    r = spend.reserve("claude-haiku-5-5", prompt_chars=3000, max_tokens=1000)  # worst case $0.0006
    spend.settle(r, "claude-haiku-5-5", input_tokens=1000, output_tokens=200)  # $0.0002
    assert abs(spend.total() - 0.0002) < 1e-9
    with pytest.raises(spend.SpendCapError):
        spend.reserve("claude-opus-5-5", prompt_chars=3000, max_tokens=1000)  # worst case $0.024 > cap
    monkeypatch.setenv("RAG_MAX_USD", "off")
    assert spend.reserve("claude-opus-5-5", prompt_chars=3000, max_tokens=1000) == 0.0
    assert spend.price("some-unknown-model") == spend.UNKNOWN_PRICE


def test_only_allowed_models(monkeypatch):
    from ragkit import llm

    monkeypatch.setenv("RAG_LLM_OFFLINE", "1")
    monkeypatch.delenv("RAG_ALLOWED_MODELS", raising=False)
    assert llm.chat("hi", model="claude-haiku-5-5") == ""
    with pytest.raises(llm.ModelNotAllowedError):
        llm.chat("hi", model="claude-opus-5-5")


def test_parallel_submissions_share_the_team_budget(tmp_path, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient

    from ragkit import spend
    from server import llm_proxy as proxy

    monkeypatch.setenv("RAG_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_MAX_USD", "off")
    monkeypatch.setenv("RAG_TEAM_MAX_USD", "0.05")
    usage = {"input_tokens": 100_000, "output_tokens": 50_000}  # $0.035 per call at Haiku prices
    prompt = "x" * 100_000 * spend.CHARS_PER_TOKEN  # so the proxy's worst-case estimate matches that
    fake = httpx.Response(200, json={"content": [], "usage": usage})
    monkeypatch.setattr(proxy._http, "post", lambda *a, **k: fake)
    client = TestClient(proxy.app)
    a = proxy.open_session(1, "team-x", 0.0, staff=False)
    b = proxy.open_session(2, "team-x", 0.0, staff=False)  # runs at the same time as a

    def call(token):
        body = {"model": "claude-haiku-5-5", "max_tokens": 50_000, "messages": [{"role": "user", "content": prompt}]}
        return client.post("/v1/messages", json=body, headers={"x-api-key": token}).status_code

    assert call(a) == 200  # team has spent $0.035 of $0.05
    assert call(b) == 403  # the parallel submission sees that spend
    assert proxy.close_session(b)["refused"]
    proxy.close_session(a)
    c = proxy.open_session(3, "team-x", 0.0, staff=False)  # database not updated yet: the live total still counts
    assert call(c) == 403
    proxy.close_session(c)
