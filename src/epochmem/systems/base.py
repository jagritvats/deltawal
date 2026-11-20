"""Shared ingestion/read machinery for all memory systems.

Write paths
  sync   : extract -> embed -> upsert, in produce order, before the query (B1).
  async  : sessions go through the broker; a worker commits facts in a (possibly
           reordered) commit order; the deterministic barrier holds the last `lag`
           commits in flight when the query fires (B2/B3/B4).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..data.stream import PreparedInstance, StreamSession
from ..llm import OllamaClient
from ..memory.extract import Fact, FactExtractor
from ..store.pg import FactStore
from ..store.wal import RedisWAL
from ..stream.bus import SessionBus
from ..stream.schedule import barrier, commit_order, low_watermark
from ..tokens import count_messages

READER_SYSTEM = (
    "You are a helpful assistant with long-term memory of past conversations with the user. "
    "Answer the user's question using the memory provided. Be concise: answer directly in one or two "
    "sentences. If the memory does not contain the information needed, say that you don't know."
)


@dataclass
class Ctx:
    cfg: Config
    client: OllamaClient
    extractor: FactExtractor
    store: FactStore
    wal: RedisWAL
    bus: SessionBus
    seed: int

    @property
    def m(self) -> dict:
        return self.cfg["models"]

    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(self.m["embedder"], texts)

    def embed_query(self, q: str) -> list[float]:
        return self.client.embed(self.m["embedder"], [self.m["query_instruction"] + q])[0]


@dataclass
class IngestResult:
    applied_epochs: list[int]                 # in commit order
    inflight_epochs: list[int]
    watermark: int | None = None              # B4 committed low-watermark
    cas_rejections: int = 0
    oocr_updated: int = 0
    oocr_corrupted: int = 0
    corrupted_keys: list[str] = field(default_factory=list)


@dataclass
class SystemOutput:
    answer: str
    messages: list[dict]
    prompt_tokens: int
    server_prompt_tokens: int | None
    retrieved: list[dict]
    extra_context: list[dict]
    ingest: IngestResult
    reader_cached: bool


def trunc(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + " [...]"


def session_message(ss: StreamSession, ns: str) -> dict:
    return {"ns": ns, "epoch": ss.epoch, "session_id": ss.session.session_id, "date": ss.session.date,
            "turns": [{"role": t.role, "content": t.content} for t in ss.session.turns]}


def compute_oocr(applied_facts: list[Fact], stored: dict[str, int]) -> tuple[int, int, list[str]]:
    """Updated keys = keys written by >=2 distinct committed epochs. A key is corrupted when its
    stored source_epoch is strictly below the newest committed epoch for that key."""
    epochs: dict[str, set[int]] = {}
    for f in applied_facts:
        epochs.setdefault(f.entity_key, set()).add(f.source_epoch)
    updated = [k for k, es in epochs.items() if len(es) >= 2]
    corrupted = [k for k in updated if stored.get(k, -1) < max(epochs[k])]
    return len(updated), len(corrupted), sorted(corrupted)


class MemorySystem(ABC):
    name: str = "abstract"
    write_mode: str = "async"    # sync | async
    policy: str = "lww"          # lww | cas
    use_wal: bool = False

    def __init__(self, ctx: Ctx):
        self.ctx = ctx

    # ------------------------------------------------------------------ ingest
    def ingest(self, prep: PreparedInstance, ns: str, lag: int, p_ooo: float) -> IngestResult:
        ctx = self.ctx
        facts_by_epoch = ctx.extractor.extract_instance(prep)
        all_facts = [f for e in sorted(facts_by_epoch) for f in facts_by_epoch[e]]
        embs = dict(zip([id(f) for f in all_facts], ctx.embed_docs([f.doc_text() for f in all_facts])))
        ctx.store.reset(ns)
        if self.use_wal:
            ctx.wal.reset(ns)

        if self.write_mode == "sync":
            applied_idx = list(range(prep.n))
            inflight_idx: list[int] = []
        else:
            msgs = []
            for ss in prep.stream:
                if self.use_wal:  # write-through WAL *before* broker publish
                    e = ctx.wal.next_epoch(ns)
                    assert e == ss.epoch, (e, ss.epoch)
                    ctx.wal.append_session(ns, e, ss)
                msgs.append(session_message(ss, ns))
            offsets = ctx.bus.publish(ns, msgs)
            received = ctx.bus.fetch(ns, offsets)
            assert [m["epoch"] for m in received] == [s.epoch for s in prep.stream]
            order = commit_order(prep.n, p_ooo, f"{ctx.seed}:{prep.inst.qid}:{p_ooo}")
            b = barrier(order, lag)
            applied_idx, inflight_idx = list(b.applied), list(b.inflight)

        rejections = 0
        applied_facts: list[Fact] = []
        wm = None
        for idx in applied_idx:
            epoch = prep.stream[idx].epoch
            for f in facts_by_epoch.get(epoch, []):
                ok = ctx.store.upsert(ns, f, embs[id(f)], self.policy)
                rejections += int(not ok)
                applied_facts.append(f)
            if self.use_wal:
                wm = ctx.wal.mark_applied(ns, epoch)
        if self.use_wal:
            assert wm is None or wm == low_watermark({prep.stream[i].epoch for i in applied_idx})
            wm = ctx.wal.committed(ns)
        n_up, n_cor, keys = compute_oocr(applied_facts, ctx.store.key_epochs(ns))
        return IngestResult(
            applied_epochs=[prep.stream[i].epoch for i in applied_idx],
            inflight_epochs=[prep.stream[i].epoch for i in inflight_idx],
            watermark=wm, cas_rejections=rejections,
            oocr_updated=n_up, oocr_corrupted=n_cor, corrupted_keys=keys,
        )

    # ------------------------------------------------------------------ read
    def retrieve_facts(self, ns: str, qemb: list[float]) -> list[dict]:
        return self.ctx.store.search(ns, qemb, self.ctx.cfg["memory"]["top_k_facts"])

    @abstractmethod
    def build_messages(self, prep: PreparedInstance, ns: str, ing: IngestResult,
                       qemb: list[float]) -> tuple[list[dict], list[dict], list[dict]]:
        """Return (messages, retrieved_facts, extra_context)."""

    def run(self, prep: PreparedInstance, ns: str, lag: int, p_ooo: float) -> SystemOutput:
        ing = self.ingest(prep, ns, lag, p_ooo)
        qemb = self.ctx.embed_query(prep.inst.question)
        messages, retrieved, extra = self.build_messages(prep, ns, ing, qemb)
        res = self.ctx.client.chat(self.ctx.m["reader"], messages, max_tokens=256)
        return SystemOutput(
            answer=res.content.strip(), messages=messages, prompt_tokens=count_messages(messages),
            server_prompt_tokens=res.prompt_eval_count, retrieved=retrieved, extra_context=extra,
            ingest=ing, reader_cached=res.cached,
        )

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def fmt_fact_plain(f: dict) -> str:
        return f"- ({f['session_date']}) {f['entity_key']}: {f['fact_text']}"

    @staticmethod
    def question_block(prep: PreparedInstance) -> str:
        return f"Current date: {prep.inst.question_date}\nQuestion: {prep.inst.question}\nAnswer:"

    def raw_turn(self, role: str, content: str) -> str:
        s = self.ctx.cfg["systems"]
        lim = s["raw_turn_user_chars"] if role == "user" else s["raw_turn_assistant_chars"]
        return f"[{role}] {trunc(content, lim)}"


def cosine_top(qemb: list[float], embs: list[list[float]], m: int) -> list[int]:
    if not embs:
        return []
    q = np.asarray(qemb, dtype=np.float32)
    E = np.asarray(embs, dtype=np.float32)
    sims = (E @ q) / (np.linalg.norm(E, axis=1) * np.linalg.norm(q) + 1e-9)
    return [int(i) for i in np.argsort(-sims, kind="stable")[:m]]
