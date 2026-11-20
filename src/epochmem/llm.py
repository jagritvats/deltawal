"""Chat + embedding clients. Every call: guard -> cache -> HTTP.

`OllamaClient` uses the native Ollama API so we can pass `think: false` and a JSON
schema in `format`. `OpenAICompatClient` exists for later cloud milestones and is
blocked by the guard unless --allow-cloud is set.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .cache import DiskCache, cache_key
from .guard import check_url


@dataclass
class ChatResult:
    content: str
    parsed: Any = None
    parse_error: str | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    duration_s: float = 0.0
    cached: bool = False
    thinking: str = ""
    done_reason: str | None = None


class CallStats:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.chat_calls = 0
        self.chat_wall_s = 0.0
        self.embed_calls = 0

    def add_chat(self, wall: float) -> None:
        with self._lock:
            self.chat_calls += 1
            self.chat_wall_s += wall


class OllamaClient:
    def __init__(self, base_url: str, cache: DiskCache, timeout: float = 900.0, num_ctx: int = 16384,
                 temperature: float = 0.0, seed: int = 42, think: bool = False):
        check_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.http = httpx.Client(timeout=timeout)
        self.defaults = {"num_ctx": num_ctx, "temperature": temperature, "seed": seed}
        self.think = think
        self.stats = CallStats()

    # ---------------- chat ----------------
    def chat(self, model: str, messages: list[dict], *, schema: dict | None = None,
             max_tokens: int = 512, options: dict | None = None, attempt: int = 0) -> ChatResult:
        opts = {**self.defaults, "num_predict": max_tokens, **(options or {})}
        if attempt:
            opts["seed"] = self.defaults["seed"] + attempt  # deterministic retry
        body: dict[str, Any] = {
            "model": model, "messages": messages, "stream": False, "think": self.think, "options": opts,
        }
        if schema is not None:
            body["format"] = schema
        key = cache_key({"kind": "chat", "endpoint": "ollama", **body})
        hit = self.cache.get(key)
        if hit is not None:
            res = self._to_result(hit, schema is not None)
            res.cached = True
            return res
        url = f"{self.base_url}/api/chat"
        check_url(url)
        t0 = time.perf_counter()
        r = self.http.post(url, json={**body, "keep_alive": "60m"})
        r.raise_for_status()
        data = r.json()
        data["_wall_s"] = time.perf_counter() - t0
        self.stats.add_chat(data["_wall_s"])
        self.cache.put(key, data, kind="chat", model=model)
        return self._to_result(data, schema is not None)

    def chat_json(self, model: str, messages: list[dict], schema: dict, max_tokens: int = 1024,
                  retries: int = 2) -> tuple[ChatResult, int]:
        """Returns (result, attempts_used). Retries use a shifted seed so they are cacheable too."""
        res: ChatResult | None = None
        for attempt in range(retries + 1):
            res = self.chat(model, messages, schema=schema, max_tokens=max_tokens, attempt=attempt)
            if res.parse_error is None:
                return res, attempt + 1
        assert res is not None
        return res, retries + 1

    @staticmethod
    def _to_result(data: dict, want_json: bool) -> ChatResult:
        msg = data.get("message", {}) or {}
        content = msg.get("content", "") or ""
        res = ChatResult(
            content=content,
            prompt_eval_count=data.get("prompt_eval_count"),
            eval_count=data.get("eval_count"),
            duration_s=float(data.get("_wall_s", 0.0)),
            thinking=msg.get("thinking", "") or "",
            done_reason=data.get("done_reason"),
        )
        if want_json:
            try:
                res.parsed = json.loads(content)
            except json.JSONDecodeError as e:
                res.parse_error = f"{e.__class__.__name__}: {e}"
        return res

    # ---------------- embeddings ----------------
    def embed(self, model: str, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float] | None] = [None] * len(texts)
        todo: list[tuple[int, str, str]] = []
        for i, t in enumerate(texts):
            key = cache_key({"kind": "embed", "endpoint": "ollama", "model": model, "input": t})
            hit = self.cache.get(key)
            if hit is not None:
                out[i] = hit
            else:
                todo.append((i, t, key))
        url = f"{self.base_url}/api/embed"
        for s in range(0, len(todo), batch_size):
            chunk = todo[s:s + batch_size]
            check_url(url)
            r = self.http.post(url, json={"model": model, "input": [c[1] for c in chunk], "keep_alive": "60m"})
            r.raise_for_status()
            vecs = r.json()["embeddings"]
            with self.stats._lock:
                self.stats.embed_calls += 1
            for (i, _, key), v in zip(chunk, vecs):
                self.cache.put(key, v, kind="embed", model=model)
                out[i] = v
        return out  # type: ignore[return-value]

    def version(self) -> str:
        url = f"{self.base_url}/api/version"
        check_url(url)
        return self.http.get(url).json().get("version", "?")

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/api/tags"
        check_url(url)
        return [m["name"] for m in self.http.get(url).json().get("models", [])]


class OpenAICompatClient:
    """For later cloud milestones; every request is guard-checked (blocked in M1)."""

    def __init__(self, base_url: str, api_key: str, cache: DiskCache, timeout: float = 300.0):
        check_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        self.http = httpx.Client(timeout=timeout, headers={"Authorization": f"Bearer {api_key}"})

    def chat(self, model: str, messages: list[dict], max_tokens: int = 512, temperature: float = 0.0,
             seed: int = 42) -> ChatResult:
        body = {"model": model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed}
        key = cache_key({"kind": "chat", "endpoint": self.base_url, **body})
        hit = self.cache.get(key)
        if hit is None:
            url = f"{self.base_url}/chat/completions"
            check_url(url)
            r = self.http.post(url, json=body)
            r.raise_for_status()
            hit = r.json()
            self.cache.put(key, hit, kind="chat", model=model)
        content = hit["choices"][0]["message"].get("content", "")
        usage = hit.get("usage", {})
        return ChatResult(content=content, prompt_eval_count=usage.get("prompt_tokens"),
                          eval_count=usage.get("completion_tokens"))
