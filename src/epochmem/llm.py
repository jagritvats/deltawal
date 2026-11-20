"""Chat + embedding clients. Every call: guard -> cache -> HTTP.

`OllamaClient` uses the native Ollama API so we can pass `think: false` and a JSON
schema in `format`. `OpenAICompatClient` exists for later cloud milestones and is
blocked by the guard unless --allow-cloud is set.
"""
from __future__ import annotations

import json
import re
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
    def __init__(self, base_url: str, cache: DiskCache | None, timeout: float = 900.0, num_ctx: int = 16384,
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
        hit = self.cache.get(key) if self.cache is not None else None
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
        if self.cache is not None:
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
            hit = self.cache.get(key) if self.cache is not None else None
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
                if self.cache is not None:
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


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_loose(text: str):
    """Parse the first JSON object in a model reply (handles code fences / prose around it)."""
    try:
        return json.loads(text), None
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text or "")
    if m:
        try:
            return json.loads(m.group(0)), None
        except json.JSONDecodeError as e:
            return None, f"JSONDecodeError: {e}"
    return None, "no JSON object found"


class OpenAICompatClient:
    """OpenAI-compatible chat client (cloud). Every request: guard -> cache -> HTTP.

    Same call surface as OllamaClient.chat / chat_json so systems and the judge are
    backend-agnostic. JSON output is requested via instructions (schema appended to the
    prompt) and parsed loosely; reasoning traces, if any, are kept out of `content`.
    """

    def __init__(self, base_url: str, api_key: str, cache: DiskCache | None, timeout: float = 300.0,
                 temperature: float = 0.0, seed: int = 42, extra_body: dict | None = None,
                 max_retries: int = 4, session_id: str = "ses_epochmem", user_agent: str = "epochmem-bench/0.1"):
        check_url(base_url)
        self.base_url = base_url.rstrip("/")
        self.cache = cache
        # Provider requirements: honest client user-agent + stable session id per run.
        self.http = httpx.Client(timeout=timeout, headers={
            "Authorization": f"Bearer {api_key}", "User-Agent": user_agent, "x-opencode-session": session_id})
        self.temperature, self.seed = temperature, seed
        self.extra_body = extra_body or {}
        self.max_retries = max_retries
        self.stats = CallStats()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def chat(self, model: str, messages: list[dict], *, schema: dict | None = None, max_tokens: int = 512,
             options: dict | None = None, attempt: int = 0) -> ChatResult:
        msgs = [dict(m) for m in messages]
        if schema is not None:
            msgs[-1]["content"] += ("\n\nRespond with ONLY a JSON object matching this JSON schema, no prose:\n"
                                    + json.dumps(schema))
        body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": self.temperature,
                "seed": self.seed + attempt, **self.extra_body}
        key = cache_key({"kind": "chat", "endpoint": self.base_url, **body})
        hit = self.cache.get(key) if self.cache is not None else None
        cached = hit is not None
        if hit is None:
            url = f"{self.base_url}/chat/completions"
            check_url(url)
            last = None
            for i in range(self.max_retries + 1):
                t0 = time.perf_counter()
                try:
                    r = self.http.post(url, json=body)
                except httpx.HTTPError as e:
                    last = repr(e)
                else:
                    if r.status_code == 200:
                        hit = r.json()
                        hit["_wall_s"] = time.perf_counter() - t0
                        break
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    if r.status_code not in (408, 409, 429, 500, 502, 503, 504):
                        raise RuntimeError(f"cloud call failed for {model}: {last}")
                time.sleep(min(30, 2 ** i))   # backoff on transient errors only (never in accuracy logic)
            if hit is None:
                raise RuntimeError(f"cloud call failed for {model} after retries: {last}")
            self.stats.add_chat(hit["_wall_s"])
            if self.cache is not None:
                self.cache.put(key, hit, kind="chat", model=model)
        msg = (hit.get("choices") or [{}])[0].get("message", {}) or {}
        usage = hit.get("usage", {}) or {}
        if not cached:
            self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += usage.get("completion_tokens", 0) or 0
        content = msg.get("content") or ""
        res = ChatResult(content=content, prompt_eval_count=usage.get("prompt_tokens"),
                         eval_count=usage.get("completion_tokens"), duration_s=float(hit.get("_wall_s", 0.0)),
                         cached=cached, thinking=msg.get("reasoning_content") or msg.get("reasoning") or "",
                         done_reason=(hit.get("choices") or [{}])[0].get("finish_reason"))
        if schema is not None:
            res.parsed, res.parse_error = parse_json_loose(content)
        return res

    def chat_json(self, model: str, messages: list[dict], schema: dict, max_tokens: int = 1024,
                  retries: int = 2) -> tuple[ChatResult, int]:
        res = None
        for attempt in range(retries + 1):
            res = self.chat(model, messages, schema=schema, max_tokens=max_tokens, attempt=attempt)
            if res.parse_error is None:
                return res, attempt + 1
        return res, retries + 1  # type: ignore[return-value]

    def list_models(self) -> list[str]:
        url = f"{self.base_url}/models"
        check_url(url)
        return [m["id"] for m in self.http.get(url).json().get("data", [])]


CLOUD_PREFIX = "opencode-go/"


class ModelRouter:
    """Routes `opencode-go/<model>` ids to the cloud client and everything else to Ollama."""

    def __init__(self, local: OllamaClient, cloud: OpenAICompatClient | None = None,
                 min_cloud_max_tokens: int = 2048):
        self.local, self.cloud = local, cloud
        self.min_cloud_max_tokens = min_cloud_max_tokens   # cloud models may spend tokens on reasoning first
        self.cache = local.cache
        self.stats = local.stats

    def _pick(self, model: str):
        if model.startswith(CLOUD_PREFIX):
            if self.cloud is None:
                raise RuntimeError(f"{model} requested but no cloud client configured (need --allow-cloud)")
            return self.cloud, model[len(CLOUD_PREFIX):]
        return self.local, model

    def _kw(self, c, kw):
        if c is self.cloud:
            kw = {**kw, "max_tokens": max(kw.get("max_tokens", 512), self.min_cloud_max_tokens)}
            kw.pop("options", None)
        return kw

    def chat(self, model, messages, **kw):
        c, m = self._pick(model)
        return c.chat(m, messages, **self._kw(c, kw))

    def chat_json(self, model, messages, schema, **kw):
        c, m = self._pick(model)
        return c.chat_json(m, messages, schema, **self._kw(c, kw))

    def embed(self, model, texts, **kw):
        return self.local.embed(model, texts, **kw)
