"""One `chat()` function for two providers: Anthropic (Claude) and OpenAI (or any OpenAI-compatible server).

    from ragkit.llm import chat
    answer = chat("What is 2+2?", system="Answer with a number only.")

Every call is counted (see `usage()`), so you can report what your pipeline costs.
Responses are cached on disk by default, so re-running an unchanged pipeline is free.
Only the models in RAG_ALLOWED_MODELS (default: claude-haiku-5-5) can be used.
A spend cap (RAG_MAX_USD, default $5, see ragkit/spend.py) refuses calls once the budget is used up.
"""

from __future__ import annotations

import json
import threading
import time

from . import config, spend, trace
from .cache import KVCache, stable_hash

class ModelNotAllowedError(ValueError):
    pass


_lock = threading.Lock()
_clients: dict[str, object] = {}
_cache: KVCache | None = None
_usage = {"calls": 0, "cached_calls": 0, "input_tokens": 0, "output_tokens": 0}
_semaphore = threading.Semaphore(int(config.get("RAG_LLM_CONCURRENCY", "8")))

# Models that accept output_config.effort, and models that support server-side refusal fallbacks.
_EFFORT_PREFIXES = ("claude-opus-5", "claude-sonnet-5", "claude-haiku-5", "claude-fable", "claude-opus-4-8", "claude-opus-4-7")
_FALLBACK_PREFIXES = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable")


def _get_cache() -> KVCache:
    global _cache
    with _lock:
        if _cache is None:
            _cache = KVCache(config.cache_dir() / "llm.sqlite")
        return _cache


def _client(provider: str):
    with _lock:
        if provider not in _clients:
            if provider == "anthropic":
                import anthropic

                socket = config.get("RAG_LLM_SOCKET")
                if socket:
                    # on the evaluation server: no network, requests go through the server's LLM proxy
                    try:  # newer SDKs are built on httpx2
                        import httpx2 as httpx
                    except ImportError:
                        import httpx

                    http = anthropic.DefaultHttpxClient(transport=httpx.HTTPTransport(uds=socket))
                    _clients[provider] = anthropic.Anthropic(max_retries=6, base_url="http://llm-proxy", http_client=http)
                else:
                    _clients[provider] = anthropic.Anthropic(max_retries=6)
            elif provider == "openai":
                import openai

                _clients[provider] = openai.OpenAI(max_retries=6)
            else:
                raise ValueError(f"unknown provider {provider!r}")
        return _clients[provider]


def _provider_for(model: str) -> str:
    if model.startswith("claude"):
        return "anthropic"
    explicit = config.get("RAG_LLM_PROVIDER")
    return explicit if explicit and explicit != "anthropic" else "openai"


def _record(input_tokens: int, output_tokens: int, cached: bool, model: str) -> None:
    with _lock:
        _usage["calls"] += 1
        if cached:
            _usage["cached_calls"] += 1
            return
        _usage["input_tokens"] += input_tokens
        _usage["output_tokens"] += output_tokens
    usage_file = config.get("RAG_USAGE_FILE")
    if usage_file:
        with _lock, open(usage_file, "a") as f:
            f.write(json.dumps({"model": model, "in": input_tokens, "out": output_tokens, "t": time.time()}) + "\n")


def usage() -> dict:
    """Token usage of all `chat()` calls in this process (cached calls cost nothing)."""
    with _lock:
        return dict(_usage)


def chat(
    prompt: str | list[dict],
    system: str | None = None,
    model: str | None = None,
    max_tokens: int = 4000,
    effort: str = "low",
    cache: bool | None = None,
) -> str:
    """Send one request and return the text of the answer.

    prompt  -- a user message string, or a full list of {"role", "content"} messages
    system  -- optional system prompt
    model   -- default: RAG_LLM_MODEL (claude-haiku-5-5); must be in RAG_ALLOWED_MODELS
    effort  -- reasoning effort for Claude models that support it ("low" | "medium" | "high")
    """
    model = model or config.llm_model()
    if model not in config.allowed_models():
        raise ModelNotAllowedError(
            f"model {model!r} is not allowed in this workshop (allowed: {', '.join(config.allowed_models())})")
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    use_cache = cache if cache is not None else config.get("RAG_LLM_CACHE", "1") == "1"
    key = stable_hash("chat", model, system, messages, max_tokens, effort)

    start = time.time()
    if config.get("RAG_LLM_OFFLINE") == "1":
        # retrieval-only runs: no request is sent and every answer is empty (only citation metrics are meaningful)
        _trace(model, system, messages, "", True, 0, 0, start)
        return ""
    if use_cache:
        hit = _get_cache().get(key)
        if hit is not None:
            _record(0, 0, True, model)
            text = hit.decode("utf-8")
            _trace(model, system, messages, text, True, 0, 0, start)
            return text

    provider = _provider_for(model)
    reserved = spend.reserve(model, len(json.dumps(messages)) + len(system or ""), max_tokens)
    try:
        with _semaphore:
            if provider == "anthropic":
                text, n_in, n_out = _chat_anthropic(model, messages, system, max_tokens, effort)
            else:
                text, n_in, n_out = _chat_openai(model, messages, system, max_tokens)
    except BaseException:
        spend.release(reserved)
        raise
    spend.settle(reserved, model, n_in, n_out)
    _record(n_in, n_out, False, model)
    if use_cache:
        _get_cache().set(key, text.encode("utf-8"))
    _trace(model, system, messages, text, False, n_in, n_out, start)
    return text


def _trace(model, system, messages, text, cached, n_in, n_out, start):
    if trace.active():
        trace.log("llm", model, model=model, system=system, messages=messages, response=text, cached=cached,
                  input_tokens=n_in, output_tokens=n_out, seconds=round(time.time() - start, 3))


def _chat_anthropic(model, messages, system, max_tokens, effort):
    client = _client("anthropic")
    kwargs = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if system:
        kwargs["system"] = system
    if model.startswith(_EFFORT_PREFIXES):
        kwargs["output_config"] = {"effort": effort}
    if model.startswith(_FALLBACK_PREFIXES):
        # If a safety classifier declines, the API retries on a fallback model inside the same call.
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
        )
    else:
        response = client.messages.create(**kwargs)
    if response.stop_reason == "refusal":
        return "", response.usage.input_tokens, response.usage.output_tokens
    text = "".join(block.text for block in response.content if block.type == "text")
    return text.strip(), response.usage.input_tokens, response.usage.output_tokens


def _chat_openai(model, messages, system, max_tokens):
    client = _client("openai")
    full = ([{"role": "system", "content": system}] if system else []) + list(messages)
    response = client.chat.completions.create(model=model, messages=full, max_completion_tokens=max_tokens)
    text = response.choices[0].message.content or ""
    u = response.usage
    return text.strip(), (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)


def chat_json(prompt: str | list[dict], **kwargs) -> dict | list | None:
    """Like chat(), but parses the first JSON object/array in the answer (None if there is none)."""
    text = chat(prompt, **kwargs)
    return parse_json(text)


def parse_json(text: str):
    """Return the first JSON object or array found in `text` (None if there is none).

    Tolerates code fences and chatter around the JSON, and stops after the first complete value,
    so two objects in a row don't break it.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
                return value
            except json.JSONDecodeError:
                continue
    return None


if __name__ == "__main__":  # quick smoke test: python -m ragkit.llm
    print(config.llm_provider(), config.llm_model())
    print(chat("Reply with exactly: pong", cache=False))
    print(usage())
