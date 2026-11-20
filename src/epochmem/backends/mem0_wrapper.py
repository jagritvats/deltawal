"""Mem0 (OSS, pinned mem0ai==1.0.11) behind the repo's guarded, SQLite-cached model clients.

Why 1.0.11: it is the last release with Mem0's classic write path (the one in the Mem0 paper): per
`add()`, one LLM call extracts facts, then one LLM call decides ADD / UPDATE / DELETE / NONE against
the top similar existing memories, and UPDATE overwrites the memory in place (the old text survives
only in the history table). mem0ai>=2.0 replaced it with an additive-only extractor (ADD + links,
no UPDATE/DELETE), which cannot answer the "does UPDATE erase history" question.

Isolation / guarantees
  * No direct provider HTTP: Mem0 is constructed with its LLM/embedder factories patched to return
    `RoutedMem0LLM` / `RoutedMem0Embedder`, which call `ModelRouter.chat` / `ModelRouter.embed`
    (guard -> SQLite cache -> HTTP; budget-capped cloud client). Mem0's own provider classes are
    never instantiated.
  * Vector store: embedded on-disk Qdrant, one directory (collection + history SQLite) per namespace
    under `store_root` (runs/mem0_qdrant/...). Nothing touches Postgres.
  * Telemetry off (MEM0_TELEMETRY=False) and MEM0_DIR redirected under runs/mem0_qdrant/.
  * Determinism: Mem0's extraction prompt embeds the wall-clock date ("Today's date is ...") at import
    time; we pin it to the session date so prompts (hence cache keys) are reproducible and the model
    resolves relative dates against the conversation date, not today.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from unittest import mock

from ..config import repo_path

os.environ["MEM0_TELEMETRY"] = "False"                      # never phone home
os.environ.setdefault("MEM0_DIR", str(repo_path("runs/mem0_qdrant/.mem0")))

from mem0.configs.base import MemoryConfig  # noqa: E402
from mem0.embeddings.base import EmbeddingBase  # noqa: E402
from mem0.llms.base import LLMBase  # noqa: E402
from mem0.memory import main as _mem0_main  # noqa: E402
from mem0.memory.utils import extract_json, remove_code_blocks  # noqa: E402

MEM0_VERSION_REQUIRED = "1.0.11"
USER_ID = "mem0_user"           # isolation is per directory (one Qdrant collection + history DB per namespace)
EVENTS = ("ADD", "UPDATE", "DELETE")
_TODAY_RE = re.compile(r"Today's date is \d{4}-\d{2}-\d{2}\.")
_DATE_RE = re.compile(r"(\d{4})[/-](\d{2})[/-](\d{2})")


def iso_date(session_date: str) -> str | None:
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20'."""
    m = _DATE_RE.search(session_date or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def trunc(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " [...]"


def session_messages(turns, session_date: str, user_chars: int = 2000, asst_chars: int = 300) -> list[dict]:
    """Mem0 input for one session: a date header + turns truncated exactly like FactExtractor
    (`render_session`: user 2000 chars, assistant 300 chars, ' [...]' marker)."""
    msgs = [{"role": "system", "content": f"SESSION DATE: {session_date}"}]
    for t in turns:
        role, content = (t["role"], t["content"]) if isinstance(t, dict) else (t.role, t.content)
        msgs.append({"role": role, "content": trunc(content, user_chars if role == "user" else asst_chars)})
    return msgs


def mem0_parse_ok(content: str, key: str) -> bool:
    """Mirror Mem0's own parsing (remove_code_blocks -> json.loads -> extract_json fallback) and
    require the expected top-level list (`facts` for extraction, `memory` for the update call)."""
    try:
        cleaned = remove_code_blocks(content or "")
        if not cleaned.strip():
            return False
        try:
            obj = json.loads(cleaned, strict=False)
        except json.JSONDecodeError:
            obj = json.loads(extract_json(content), strict=False)
        return isinstance(obj, dict) and isinstance(obj.get(key), list)
    except Exception:
        return False


@dataclass
class InternalStats:
    calls: int = 0
    live_calls: int = 0
    cached_calls: int = 0
    parse_failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    live_wall_s: float = 0.0
    by_kind: dict = field(default_factory=lambda: {"extract": [0, 0], "update": [0, 0]})  # [calls, parse_fail]
    embed_calls: int = 0
    mem0_errors: int = 0            # ERROR records logged by mem0 (bad JSON, hallucinated ids, ...)
    failure_samples: list = field(default_factory=list)

    def snapshot(self) -> dict:
        d = asdict(self)
        d["by_kind"] = {k: list(v) for k, v in self.by_kind.items()}
        d["failure_samples"] = list(self.failure_samples)
        return d

    @staticmethod
    def delta(after: dict, before: dict) -> dict:
        out = {}
        for k, v in after.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[k] = round(v - before.get(k, 0), 4) if isinstance(v, float) else v - before.get(k, 0)
            elif k == "by_kind":
                out[k] = {kk: [vv[0] - before[k][kk][0], vv[1] - before[k][kk][1]] for kk, vv in v.items()}
        out["failure_samples"] = after["failure_samples"][len(before["failure_samples"]):]
        return out


class RoutedMem0LLM(LLMBase):
    """Mem0 LLM whose every call goes through ModelRouter.chat (guard -> cache -> budget -> HTTP)."""

    def __init__(self, router, model: str, stats: InternalStats, max_tokens: int = 2048):
        from mem0.configs.llms.base import BaseLlmConfig
        super().__init__(BaseLlmConfig(model=model, temperature=0.0, max_tokens=max_tokens))
        self.router, self.model, self.stats, self.max_tokens = router, model, stats, max_tokens
        self.session_date: str | None = None          # ISO date of the session being ingested
        self._lock = threading.Lock()

    def generate_response(self, messages, response_format=None, tools=None, tool_choice="auto", **kw):
        if tools:
            raise NotImplementedError("tool calls are not used by Mem0's vector-store path")
        msgs = [dict(m) for m in messages]
        kind = "update" if any("smart memory manager" in str(m.get("content", "")) for m in msgs) else "extract"
        if kind == "extract" and self.session_date:
            for m in msgs:
                m["content"] = _TODAY_RE.sub(f"Today's date is {self.session_date}.", str(m["content"]))
        res = self.router.chat(self.model, msgs, max_tokens=self.max_tokens)
        ok = mem0_parse_ok(res.content, "facts" if kind == "extract" else "memory")
        with self._lock:
            s = self.stats
            s.calls += 1
            s.by_kind[kind][0] += 1
            s.cached_calls += int(res.cached)
            s.live_calls += int(not res.cached)
            s.live_wall_s += 0.0 if res.cached else res.duration_s
            s.prompt_tokens += res.prompt_eval_count or 0
            s.completion_tokens += res.eval_count or 0
            if not ok:
                s.parse_failures += 1
                s.by_kind[kind][1] += 1
                if len(s.failure_samples) < 20:
                    s.failure_samples.append({"kind": kind, "done_reason": res.done_reason,
                                              "content": (res.content or "")[:300]})
        return res.content


class RoutedMem0Embedder(EmbeddingBase):
    """Mem0 embedder over ModelRouter.embed (Ollama, cached). Search queries get the embedder's
    instruction prefix; stored memories do not (same convention as the rest of the repo)."""

    def __init__(self, router, model: str, query_instruction: str, stats: InternalStats, dim: int):
        from mem0.configs.embeddings.base import BaseEmbedderConfig
        super().__init__(BaseEmbedderConfig(model=model, embedding_dims=dim))
        self.router, self.model, self.qi, self.stats = router, model, query_instruction, stats

    def embed(self, text, memory_action=None):
        self.stats.embed_calls += 1
        t = (self.qi + text) if memory_action == "search" else text
        return list(map(float, self.router.embed(self.model, [t])[0]))

    def embed_batch(self, texts, memory_action=None):
        return [self.embed(t, memory_action) for t in texts]


class _ErrorCounter(logging.Handler):
    def __init__(self, stats: InternalStats):
        super().__init__(level=logging.ERROR)
        self.stats = stats

    def emit(self, record):
        self.stats.mem0_errors += 1


_CONSTRUCT_LOCK = threading.Lock()


@contextmanager
def _routed_factories(llm, embedder):
    """Make Memory.__init__ use our routed LLM/embedder instead of instantiating provider SDK clients."""
    with _CONSTRUCT_LOCK, \
            mock.patch.object(_mem0_main.LlmFactory, "create", lambda *a, **k: llm), \
            mock.patch.object(_mem0_main.EmbedderFactory, "create", lambda *a, **k: embedder):
        yield


class Mem0Backend:
    """One Mem0 memory per namespace (directory = Qdrant collection + history SQLite).

    API: reset(ns), add_session(ns, turns, epoch, session_date), search(ns, query, k),
    history_counts(ns), history_rows(ns), snapshot(src_ns, dst_ns), close(ns).
    """

    def __init__(self, router, llm_model: str, embed_model: str, query_instruction: str, dim: int,
                 store_root: str | Path, user_chars: int = 2000, asst_chars: int = 300, max_tokens: int = 2048):
        import importlib.metadata as md
        ver = md.version("mem0ai")
        if ver != MEM0_VERSION_REQUIRED:
            raise RuntimeError(f"mem0ai=={MEM0_VERSION_REQUIRED} required (classic ADD/UPDATE/DELETE path), got {ver}")
        self.version = ver
        self.router, self.llm_model, self.embed_model = router, llm_model, embed_model
        self.qi, self.dim = query_instruction, dim
        self.root = Path(store_root)
        self.user_chars, self.asst_chars = user_chars, asst_chars
        self.stats = InternalStats()
        self.llm = RoutedMem0LLM(router, llm_model, self.stats, max_tokens=max_tokens)
        self.embedder = RoutedMem0Embedder(router, embed_model, query_instruction, self.stats, dim)
        self._open: dict[str, object] = {}
        lg = logging.getLogger("mem0")
        lg.setLevel(logging.WARNING)
        if not any(isinstance(h, _ErrorCounter) for h in lg.handlers):
            lg.addHandler(_ErrorCounter(self.stats))

    # ------------------------------------------------------------ storage
    def path(self, ns: str) -> Path:
        return self.root / re.sub(r"[^A-Za-z0-9_.-]+", "_", ns)

    def _mem(self, ns: str):
        if ns in self._open:
            return self._open[ns]
        d = self.path(ns)
        d.mkdir(parents=True, exist_ok=True)
        cfg = MemoryConfig(
            vector_store={"provider": "qdrant", "config": {"collection_name": "mem0", "embedding_model_dims": self.dim,
                                                           "path": str(d / "qdrant"), "on_disk": True}},
            llm={"provider": "ollama", "config": {"model": self.llm_model}},           # placeholder: never built
            embedder={"provider": "ollama", "config": {"model": self.embed_model}},    # placeholder: never built
            history_db_path=str(d / "history.db"),
        )
        with _routed_factories(self.llm, self.embedder):
            m = _mem0_main.Memory(cfg)
        assert m.llm is self.llm and m.embedding_model is self.embedder, "Mem0 must use the routed clients"
        self._open[ns] = m
        return m

    def close(self, ns: str) -> None:
        m = self._open.pop(ns, None)
        if m is not None:
            try:
                m.vector_store.client.close()
            finally:
                m.close()

    def close_all(self) -> None:
        for ns in list(self._open):
            self.close(ns)

    def reset(self, ns: str) -> None:
        self.close(ns)
        shutil.rmtree(self.path(ns), ignore_errors=True)

    def snapshot(self, src_ns: str, dst_ns: str) -> None:
        """Copy a namespace's on-disk state (Qdrant + history) so it can keep evolving separately."""
        self.close(src_ns)
        self.reset(dst_ns)
        shutil.copytree(self.path(src_ns), self.path(dst_ns), ignore=shutil.ignore_patterns("meta.json"))

    def exists(self, ns: str) -> bool:
        return (self.path(ns) / "history.db").exists()

    # ------------------------------------------------------------ write
    def _max_rowid(self, m) -> int:
        with m.db._lock:
            return int(m.db.connection.execute("SELECT COALESCE(MAX(rowid), 0) FROM history").fetchone()[0])

    def add_session(self, ns: str, session_turns, epoch: int, session_date: str, session_id: str | None = None) -> dict:
        """Ingest one session through Mem0's full inference path. Returns per-session stats and the
        history rows (authoritative ADD/UPDATE/DELETE with old -> new text) this session produced."""
        m = self._mem(ns)
        before, row0, t0 = self.stats.snapshot(), self._max_rowid(m), time.perf_counter()
        self.llm.session_date = iso_date(session_date)
        try:
            res = m.add(session_messages(session_turns, session_date, self.user_chars, self.asst_chars),
                        user_id=USER_ID,
                        metadata={"source_epoch": int(epoch), "session_date": session_date,
                                  **({"session_id": session_id} if session_id else {})})
        finally:
            self.llm.session_date = None
        wall = time.perf_counter() - t0
        with m.db._lock:
            rows = m.db.connection.execute(
                "SELECT rowid, memory_id, old_memory, new_memory, event FROM history WHERE rowid > ? ORDER BY rowid",
                (row0,)).fetchall()
        events = [{"memory_id": r[1], "old": r[2], "new": r[3], "event": r[4], "epoch": int(epoch)} for r in rows]
        counts = {e: sum(x["event"] == e for x in events) for e in EVENTS}
        n_none = sum(1 for r in (res or {}).get("results", []) if r.get("event") == "NONE")
        return {"epoch": int(epoch), "session_date": session_date, "wall_s": round(wall, 3),
                "n_turns": len(session_turns), "events": counts, "n_none_returned": n_none,
                "history": events, "internal": InternalStats.delta(self.stats.snapshot(), before)}

    # ------------------------------------------------------------ read
    def search(self, ns: str, query: str, k: int) -> list[dict]:
        m = self._mem(ns)
        out = []
        for r in m.search(query, user_id=USER_ID, limit=k).get("results", []):
            md = r.get("metadata") or {}
            out.append({"id": r["id"], "memory": r["memory"], "score": float(r.get("score") or 0.0),
                        "source_epoch": md.get("source_epoch"), "session_date": md.get("session_date"),
                        "metadata": md})
        return out

    def history_rows(self, ns: str) -> list[dict]:
        m = self._mem(ns)
        with m.db._lock:
            ids = [r[0] for r in m.db.connection.execute(
                "SELECT memory_id FROM history GROUP BY memory_id ORDER BY MIN(rowid)").fetchall()]
        return [{"memory_id": mid, "history": m.history(mid)} for mid in ids]

    def history_counts(self, ns: str) -> dict:
        """Totals of Mem0 history events (via Memory.history()) plus per-memory counts."""
        per = {}
        tot = {e: 0 for e in EVENTS}
        for item in self.history_rows(ns):
            c = {e: 0 for e in EVENTS}
            for h in item["history"]:
                if h["event"] in c:
                    c[h["event"]] += 1
                    tot[h["event"]] += 1
            per[item["memory_id"]] = c
        live = sum(1 for c in per.values() if c["DELETE"] == 0)
        return {"totals": tot, "n_memories_ever": len(per), "n_live": live,
                "n_updated_memories": sum(1 for c in per.values() if c["UPDATE"] > 0), "per_memory": per}


# ---------------------------------------------------------------- reader-side rendering (pure, unit-tested)
def memory_line(m: dict, label: str) -> str:
    return f"[{label} | epoch={m['source_epoch']} | {m['session_date']}] {m['memory']}"


def wal_turn_line(t: dict, user_chars: int, asst_chars: int) -> str:
    txt = trunc(t["content"], user_chars if t["role"] == "user" else asst_chars)
    return f"[NEWER MESSAGE | epoch={t['epoch']} | {t['date']}] [{t['role']}] {txt}"


def assemble_lines(memories: list[dict], label: str, wal_turns: list[dict] | None = None,
                   user_chars: int = 2000, asst_chars: int = 600) -> list[str]:
    """Chronological (epoch) order like deltawal `_assemble`: memories before WAL turns of the same
    epoch; ties among memories keep Mem0's score order (stable sort)."""
    items = [((m["source_epoch"] if m["source_epoch"] is not None else -1), 0, memory_line(m, label)) for m in memories]
    items += [(t["epoch"], 1, wal_turn_line(t, user_chars, asst_chars)) for t in (wal_turns or [])]
    return [x[2] for x in sorted(items, key=lambda x: (x[0], x[1]))]


def reader_messages(lines: list[str], question_block: str, reader_system: str, precedence: str | None) -> list[dict]:
    user = "Memory:\n" + ("\n".join(lines) or "(none)") + f"\n\n{question_block}"
    sys = reader_system + ("\n\n" + precedence if precedence else "")
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]
