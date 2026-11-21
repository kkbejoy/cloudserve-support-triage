"""The only place that calls a language model (B-08; NFR-02, NFR-07, A11).

    llm = get_llm()
    reply = llm.complete(prompt)          # -> LLMReply(text, model, cached, latency_ms)
                                          # raises LLMUnavailable when the provider can't answer

- Groq or OpenRouter, both OpenAI-compatible (LLM_PROVIDER, MODEL_NAME, *_API_KEY in .env).
- temperature 0, so the same prompt gives the same answer as far as the provider allows.
- Cache: every reply is stored by a hash of (model, prompt, settings); a re-run costs no requests
  and is reproducible.
- Retries with exponential backoff on timeouts, 429 and 5xx; waits between calls to stay inside
  free-tier per-minute limits.
- Circuit breaker: after LLM_CIRCUIT_BREAKER consecutive failures the provider is treated as down
  for the rest of the run, so an outage escalates tickets quickly instead of hanging.
The API key is read from the environment only and never logged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)

PROVIDERS = {
    "groq": {"url": "https://api.groq.com/openai/v1/chat/completions", "key": "GROQ_API_KEY",
             "default_model": "openai/gpt-oss-20b", "min_interval_s": 2.1},
    "openrouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "key": "OPENROUTER_API_KEY",
                   "default_model": "meta-llama/llama-3.1-8b-instruct", "min_interval_s": 3.1},
}


class LLMUnavailable(RuntimeError):
    """The model could not produce an answer (no key, outage, rate limit, repeated errors)."""


class LLMReply(BaseModel):
    text: str
    model: str
    cached: bool = False
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, max_tokens: int = 600) -> LLMReply:
        ...


class _Cache:
    def __init__(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, text TEXT, model TEXT,"
                           " created_at TEXT)")
        self._conn.commit()

    def get(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT text FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put(self, key: str, text: str, model: str) -> None:
        self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, datetime('now'))", (key, text, model))
        self._conn.commit()


def _key_from_env(name: str) -> str:
    """API key from the environment; the .env.example placeholder counts as not set."""
    key = os.getenv(name, "").strip()
    return "" if key in ("", "your_key_here") else key


class OpenAICompatibleLLM:
    """Groq / OpenRouter client with cache, retries, pacing and a circuit breaker."""

    def __init__(self, settings: Optional[Settings] = None, provider: Optional[str] = None,
                 use_cache: bool = True, model: Optional[str] = None) -> None:
        import requests  # in requirements.txt; imported here so tests without network don't need it

        self._requests = requests
        s = settings or get_settings()
        prov = provider or s.llm_provider
        if prov == "auto":
            prov = "groq" if _key_from_env("GROQ_API_KEY") else "openrouter"
        if prov not in PROVIDERS:
            raise ValueError(f"Unknown LLM_PROVIDER {prov!r}; use groq, openrouter or auto")
        cfg = PROVIDERS[prov]
        self.provider, self.url = prov, cfg["url"]
        self._key = _key_from_env(cfg["key"])
        self.model = model or s.model_name or cfg["default_model"]
        self.name = f"{prov}:{self.model}"
        self.timeout, self.max_retries = s.llm_timeout_s, s.llm_max_retries
        self.min_interval = float(s.llm_min_interval_s) if s.llm_min_interval_s else cfg["min_interval_s"]
        self.breaker_limit = s.llm_circuit_breaker
        self._cache = _Cache(s.llm_cache_path) if use_cache else None
        self._last_call = 0.0
        self._consecutive_failures = 0
        self.calls = self.cache_hits = self.failures = 0
        self.tokens_in = self.tokens_out = 0          # as reported by the provider (daily limits)

    def _body(self, prompt: str, max_tokens: int) -> dict:
        body = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if "gpt-oss" in self.model:
            # reasoning model: its hidden reasoning counts against max_tokens, so keep it short
            # and leave room for the JSON answer
            body["reasoning_effort"] = "low"
            body["max_tokens"] = max_tokens + 800
        return body

    def check(self) -> None:
        """One tiny call before a run. Raises LLMUnavailable if the provider rejects the setup
        (bad key, unknown model) so a misconfigured run stops instead of escalating everything."""
        self.complete("Reply with the single word OK.", max_tokens=5)

    @property
    def available(self) -> bool:
        return bool(self._key) and self._consecutive_failures < self.breaker_limit

    def _cache_key(self, prompt: str, max_tokens: int) -> str:
        blob = json.dumps({"model": self.name, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0})
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def complete(self, prompt: str, max_tokens: int = 600) -> LLMReply:
        key = self._cache_key(prompt, max_tokens)
        if self._cache:
            hit = self._cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return LLMReply(text=hit, model=self.name, cached=True)
        if not self._key:
            raise LLMUnavailable(f"No API key set for {self.provider} ({PROVIDERS[self.provider]['key']})")
        if self._consecutive_failures >= self.breaker_limit:
            raise LLMUnavailable(f"{self.provider} treated as unavailable after "
                                 f"{self._consecutive_failures} consecutive failures")
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            t0 = time.perf_counter()
            self._last_call = time.monotonic()
            self.calls += 1
            try:
                resp = self._requests.post(
                    self.url, timeout=self.timeout,
                    headers={"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"},
                    json=self._body(prompt, max_tokens))
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    usage = data.get("usage") or {}
                    t_in, t_out = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
                    self.tokens_in += t_in
                    self.tokens_out += t_out
                    self._consecutive_failures = 0
                    if self._cache:
                        self._cache.put(key, text, self.name)
                    return LLMReply(text=text, model=self.name, latency_ms=(time.perf_counter() - t0) * 1000,
                                    tokens_in=t_in, tokens_out=t_out)
                last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
                if resp.status_code in (400, 401, 402, 403, 404):   # will not succeed on retry
                    break
                retry_after = resp.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else 2 ** attempt
            except Exception as exc:  # timeout, connection error, malformed response
                last_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                delay = 2 ** attempt
            if attempt < self.max_retries:
                time.sleep(min(delay, 20))
        self.failures += 1
        self._consecutive_failures += 1
        logger.warning("LLM call failed (%s); consecutive failures %d", last_error, self._consecutive_failures)
        raise LLMUnavailable(last_error)


_default: Optional[OpenAICompatibleLLM] = None


def get_llm() -> OpenAICompatibleLLM:
    """One shared client per process (keeps the pacing and circuit-breaker state)."""
    global _default
    if _default is None:
        _default = OpenAICompatibleLLM()
    return _default
