"""Embeddings with an on-disk cache.

    from ragkit.embed import embed
    vectors = embed(["first chunk", "second chunk"])            # default model (RAG_EMBED_MODEL)
    vectors = embed(texts, model="openai:text-embedding-3-small")
    vectors = embed(texts, model="local:BAAI/bge-small-en-v1.5")
    vectors = embed(texts, model="local+trust:jinaai/jina-embeddings-v2-base-code")  # runs the model repo's code

Returns a float32 numpy array of shape (len(texts), dim), L2-normalised,
so cosine similarity is just a dot product:  scores = doc_vectors @ query_vector
"""

from __future__ import annotations

import threading
import time

import numpy as np

from . import config
from . import trace
from .cache import KVCache, stable_hash

_lock = threading.Lock()
_models: dict[str, object] = {}
_cache: KVCache | None = None


def _get_cache() -> KVCache:
    global _cache
    with _lock:
        if _cache is None:
            _cache = KVCache(config.cache_dir() / "embeddings.sqlite")
        return _cache


def _local_model(name: str, trust_remote_code: bool = False):
    with _lock:
        if name not in _models:
            from sentence_transformers import SentenceTransformer

            _models[name] = SentenceTransformer(name, device="cpu", trust_remote_code=trust_remote_code)
        return _models[name]


def _embed_uncached(texts: list[str], model: str, batch_size: int) -> np.ndarray:
    provider, _, name = model.partition(":")
    if provider in ("local", "local+trust"):
        # "local+trust:" opts in to executing Python code shipped with the model repo (needed by some code models)
        st = _local_model(name, trust_remote_code=provider == "local+trust")
        return st.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 2000)
    if provider == "openai":
        import openai

        client = openai.OpenAI()
        out = []
        for i in range(0, len(texts), 256):
            response = client.embeddings.create(model=name, input=[t[:30000] or " " for t in texts[i : i + 256]])
            out.extend(d.embedding for d in response.data)
        return np.asarray(out, dtype=np.float32)
    raise ValueError(f"unknown embedding provider in {model!r}; use 'local:<name>' or 'openai:<name>'")


def embed(texts: list[str], model: str | None = None, batch_size: int = 64) -> np.ndarray:
    model = model or config.embed_model()
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    start = time.time()
    cache = _get_cache()
    keys = [stable_hash("emb", model, t) for t in texts]
    found = cache.get_many(list(set(keys)))

    missing_order: dict[str, str] = {}
    for k, t in zip(keys, texts):
        if k not in found and k not in missing_order:
            missing_order[k] = t
    if missing_order:
        miss_keys, miss_texts = list(missing_order), list(missing_order.values())
        for i in range(0, len(miss_texts), 4096):
            vecs = np.asarray(_embed_uncached(miss_texts[i : i + 4096], model, batch_size), dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-12)
            new = {k: v.tobytes() for k, v in zip(miss_keys[i : i + 4096], vecs)}
            cache.set_many(new)
            found.update(new)
    vectors = np.stack([np.frombuffer(found[k], dtype=np.float32) for k in keys])
    if trace.active():
        trace.log("embed", f"embedded {len(texts)} text(s)", model=model, n=len(texts), computed=len(missing_order),
                  dim=int(vectors.shape[1]), seconds=round(time.time() - start, 3), texts=texts[:5],
                  vector_preview=[round(float(x), 3) for x in vectors[0][:64]])
    return vectors


def embed_one(text: str, model: str | None = None) -> np.ndarray:
    return embed([text], model=model)[0]
