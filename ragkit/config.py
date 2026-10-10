"""Configuration: everything comes from environment variables (or a .env file).

Variables
---------
RAG_LLM_PROVIDER     "anthropic" | "openai"   (default: whichever key is set, anthropic first)
RAG_LLM_MODEL        default chat model        (default: claude-haiku-5-5 / gpt-4.1-mini)
RAG_ALLOWED_MODELS   comma-separated chat models chat() accepts (default: claude-haiku-5-5; Opus is too expensive)
RAG_EMBED_MODEL      default embedding model   (default: local:sentence-transformers/all-MiniLM-L6-v2)
RAG_CACHE_DIR        where embeddings / LLM responses are cached (default: ~/.cache/ragkit)
RAG_LLM_CACHE        "1" to cache LLM responses on disk (default: "1")
RAG_LLM_OFFLINE      "1": chat() sends nothing and returns "" (retrieval-only runs without any API cost)
RAG_MAX_USD          spend cap in USD for LLM calls through ragkit, or "off" (default: 5; see ragkit/spend.py)
RAG_EMBED_SHARED_CACHE  extra read-only embeddings cache (sqlite) to look in first (the server's sandbox uses it)
RAG_SERVER           evaluation server URL for scripts/submit.py (default: https://viscon-eval.lukasre.ch)
RAG_ENV_FILE         extra .env file to load (e.g. a shared key file)
OPENAI_API_KEY, OPENAI_BASE_URL, ANTHROPIC_API_KEY   provider credentials
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_SERVER = "https://viscon-eval.lukasre.ch"
_loaded = False


def load_env() -> None:
    global _loaded
    if _loaded:
        return
    load_dotenv(Path.cwd() / ".env", override=False)
    extra = os.environ.get("RAG_ENV_FILE")
    if extra:
        load_dotenv(extra, override=False)
    _loaded = True


def get(name: str, default: str | None = None) -> str | None:
    load_env()
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def cache_dir() -> Path:
    path = Path(get("RAG_CACHE_DIR", str(Path.home() / ".cache" / "ragkit")))
    path.mkdir(parents=True, exist_ok=True)
    return path


def llm_provider() -> str:
    explicit = get("RAG_LLM_PROVIDER")
    if explicit:
        return explicit
    if get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


# Cheap, fast models by default: a full test run is cents, not dollars. Set RAG_LLM_MODEL for a bigger one.
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4.1-mini"}


def llm_model() -> str:
    return get("RAG_LLM_MODEL", DEFAULT_MODELS[llm_provider()])


def allowed_models() -> list[str]:
    """Chat models `chat()` will call. Only Haiku by default: a full test run with Opus costs dollars, not cents."""
    return [m.strip() for m in get("RAG_ALLOWED_MODELS", "claude-haiku-5-5").split(",") if m.strip()]


def model_tag(model: str | None = None) -> str:
    """Short name of a model for file names and labels: claude-haiku-5-5 -> haiku, gpt-4.1-mini -> gpt-4.1-mini."""
    model = model or llm_model()
    return model.split("-")[1] if model.startswith("claude-") else model


def embed_model() -> str:
    return get("RAG_EMBED_MODEL", "local:sentence-transformers/all-MiniLM-L6-v2")
