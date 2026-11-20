"""WAL turn/sentence embedding, off the write-ack path.

Each WAL turn gets (a) one turn-level embedding of its (truncated) text, used for relevance
gating, and (b) one embedding per sentence, used by sentence-level shadow masking. A
background worker drains the `pending` queue; the read path lazily embeds anything the
worker has not reached yet, so correctness never depends on background progress.
"""
from __future__ import annotations

import re
import threading

import numpy as np

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(text: str, max_sents: int = 24, min_chars: int = 4, max_chars: int = 400) -> list[str]:
    out = []
    for s in _SPLIT.split(text or ""):
        s = s.strip()
        if len(s) >= min_chars:
            out.append(s[:max_chars])
        if len(out) >= max_sents:
            break
    return out or ([text.strip()[:max_chars]] if text and text.strip() else [])


def turn_text(role: str, content: str, sys_cfg: dict) -> str:
    lim = sys_cfg["raw_turn_user_chars"] if role == "user" else sys_cfg["raw_turn_assistant_chars"]
    return content if len(content) <= lim else content[:lim] + " [...]"


def embed_records(embed_docs, recs: list[dict], sys_cfg: dict, dim: int) -> list[tuple]:
    """Embed turns + sentences for WAL records in one batched call. Returns store_embeddings items
    and attaches `embedding` / `sent_embeddings` to each record in place."""
    texts, spans = [], []
    for r in recs:
        tt = turn_text(r["role"], r["content"], sys_cfg)
        sents = split_sentences(tt)
        spans.append((len(texts), len(sents)))
        texts.append(tt)
        texts.extend(sents)
    if not texts:
        return []
    vecs = embed_docs(texts)
    items = []
    for r, (start, k) in zip(recs, spans):
        temb = np.asarray(vecs[start], dtype=np.float32)
        smat = np.asarray(vecs[start + 1:start + 1 + k], dtype=np.float32).reshape(-1, dim)
        r["embedding"], r["sent_embeddings"] = temb, smat
        items.append((r["epoch"], r["turn"], temb, smat))
    return items


def embed_turns_only(embed_docs, recs: list[dict], sys_cfg: dict) -> list[tuple]:
    """Turn-level vectors only (enough for relevance gating). Cheap read-time lazy fill."""
    if not recs:
        return []
    vecs = embed_docs([turn_text(r["role"], r["content"], sys_cfg) for r in recs])
    items = []
    for r, v in zip(recs, vecs):
        r["embedding"] = np.asarray(v, dtype=np.float32)
        items.append((r["epoch"], r["turn"], r["embedding"], None))
    return items


def embed_sentences_only(embed_docs, recs: list[dict], sys_cfg: dict, dim: int) -> list[tuple]:
    """Sentence vectors for the given (gated) turns only."""
    texts, spans = [], []
    for r in recs:
        sents = split_sentences(turn_text(r["role"], r["content"], sys_cfg))
        spans.append((len(texts), len(sents)))
        texts.extend(sents)
    vecs = embed_docs(texts) if texts else []
    items = []
    for r, (start, k) in zip(recs, spans):
        r["sent_embeddings"] = np.asarray(vecs[start:start + k], dtype=np.float32).reshape(-1, dim)
        items.append((r["epoch"], r["turn"], None, r["sent_embeddings"]))
    return items


class BackgroundEmbedder(threading.Thread):
    """Drains wal:{ns}:pending and stores turn + sentence embeddings."""

    def __init__(self, wal, ns: str, embed_docs, sys_cfg: dict, dim: int):
        super().__init__(daemon=True)
        self.wal, self.ns, self.embed_docs, self.sys_cfg, self.dim = wal, ns, embed_docs, sys_cfg, dim
        self.stop_evt = threading.Event()
        self.embedded_sessions = 0

    def run(self):
        while not self.stop_evt.is_set():
            e = self.wal.pop_pending(self.ns, timeout=0.2)
            if e is None:
                continue
            recs = self.wal.session_turns(self.ns, e)
            self.wal.store_embeddings(self.ns, embed_records(self.embed_docs, recs, self.sys_cfg, self.dim))
            self.embedded_sessions += 1

    def stop(self):
        self.stop_evt.set()
