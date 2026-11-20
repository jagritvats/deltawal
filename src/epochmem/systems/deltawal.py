"""DeltaWAL family: one write path, three read paths.

Write path (shared, identical across variants)
  Every turn is written through to a Redis WAL with a monotonic epoch and its embedding
  *before* the Kafka publish. The async worker commits extracted facts to PostgreSQL with
  CAS epoch fencing (an upsert lands only if its source_epoch is newer) and advances the
  contiguous committed_epoch watermark.

Read paths
  deltawal_prompt : committed top-k facts + relevance-gated uncommitted WAL turns, with
                    neutral chronological tags and an epoch-precedence instruction.
                    No extra model calls.
  deltawal_shadow : as prompt, but a committed fact is masked when it is semantically
                    shadowed by a newer gated WAL turn (cosine >= shadow_threshold),
                    using embeddings computed at write time. No query-time LLM calls.
  deltawal_lazy   : query-time extraction over ONLY the gated WAL turns, merged by
                    entity_key/epoch against the committed facts (Pareto upper bound;
                    one extra LLM call on the read path).
"""
from __future__ import annotations

import time

import numpy as np

from ..memory.extract import FACT_SCHEMA, SYSTEM_PROMPT, normalize_key
from ..memory.wal_embed import embed_sentences_only, embed_turns_only
from .base import READER_SYSTEM, MemorySystem

LAZY_SCHEMA = {
    "type": "object",
    "properties": {
        "user_mentions": {"type": "array", "items": {"type": "string"}},
        "facts": {"type": "array", "items": {
            "type": "object",
            "properties": {**FACT_SCHEMA["properties"]["facts"]["items"]["properties"], "epoch": {"type": "integer"}},
            "required": ["entity_key", "fact", "event_date", "epoch"],
        }},
    },
    "required": ["user_mentions", "facts"],
}

PRECEDENCE = (
    "Memory items are tagged with an epoch: the write sequence number of the conversation they came from "
    "(higher = more recent). If two items disagree about the same thing, the item with the higher epoch is "
    "the current one."
)


def _unit(m: np.ndarray) -> np.ndarray:
    m = np.atleast_2d(np.asarray(m, dtype=np.float32))
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def gate_wal(qemb, turns: list[dict], threshold: float, top_m: int) -> list[dict]:
    """Keep uncommitted WAL turns whose cosine similarity to the query is >= threshold
    (at most top_m, most similar first; returned in chronological order). Each returned turn
    gets a `sim` field. Empty input or nothing above threshold -> []."""
    cand = [t for t in turns if t.get("embedding") is not None]
    if not cand:
        return []
    sims = (_unit(np.stack([t["embedding"] for t in cand])) @ _unit(qemb)[0])
    keep = [i for i in np.argsort(-sims, kind="stable") if sims[i] >= threshold][:top_m]
    out = [{**cand[i], "sim": float(sims[i])} for i in keep]
    return sorted(out, key=lambda t: (t["epoch"], t["turn"]))


def shadow_mask(facts: list[dict], turns: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    """Split facts into (kept, suppressed). A committed fact is suppressed when ANY newer
    (higher-epoch) gated WAL turn has cosine similarity >= threshold with it.
    Facts need `embedding` and `source_epoch`; turns need `embedding` and `epoch`."""
    tv = [t for t in turns if t.get("embedding") is not None]
    if not facts or not tv:
        return list(facts), []
    F = _unit(np.stack([f["embedding"] for f in facts]))
    T = _unit(np.stack([t["embedding"] for t in tv]))
    S = F @ T.T
    kept, sup = [], []
    for i, f in enumerate(facts):
        newer = [j for j, t in enumerate(tv) if t["epoch"] > f["source_epoch"]]
        best = max(newer, key=lambda j: S[i, j], default=None)
        if best is not None and S[i, best] >= threshold:
            sup.append({**f, "shadowed_by": (tv[best]["epoch"], tv[best]["turn"]), "shadow_sim": float(S[i, best])})
        else:
            kept.append(f)
    return kept, sup


def shadow_mask_v2(facts: list[dict], turns: list[dict], threshold: float) -> tuple[list[dict], list[dict]]:
    """Sentence-level shadow: a committed fact is suppressed when its max cosine similarity to ANY
    sentence of ANY newer (higher-epoch) gated WAL turn is >= threshold. Turns need
    `sent_embeddings` [k, d] (falls back to the turn embedding when absent)."""
    if not facts or not turns:
        return list(facts), []
    F = _unit(np.stack([f["embedding"] for f in facts]))
    kept, sup = [], []
    per_turn = []
    for t in turns:
        S = t.get("sent_embeddings")
        if S is None or len(S) == 0:
            S = t.get("embedding")
        if S is None:
            continue
        per_turn.append((t, F @ _unit(S).T))          # [n_facts, k_sentences]
    for i, f in enumerate(facts):
        best, best_sim, best_sent = None, -1.0, None
        for t, M in per_turn:
            if t["epoch"] > f["source_epoch"]:
                j = int(np.argmax(M[i]))
                if M[i, j] > best_sim:
                    best, best_sim, best_sent = t, float(M[i, j]), j
        if best is not None and best_sim >= threshold:
            sup.append({**f, "shadowed_by": (best["epoch"], best["turn"], best_sent), "shadow_sim": best_sim})
        else:
            kept.append(f)
    return kept, sup


def strip_emb(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("embedding", "sent_embeddings")}


class _DeltaWALBase(MemorySystem):
    write_mode = "async"
    policy = "cas"
    use_wal = True
    needs_sentences = False

    def _common(self, ns, qemb):
        s = self.ctx.cfg["systems"]
        facts = self.ctx.store.search(ns, qemb, self.ctx.cfg["memory"]["top_k_facts"], with_embedding=True)
        committed, delta = self.ctx.wal.snapshot(ns)     # single Redis round trip
        # Lazy fill (query arrived before the background embedder): turn vectors for gating only,
        # then sentence vectors only for the few gated turns that still lack them.
        missing = [t for t in delta if t.get("embedding") is None]
        if missing:
            self.ctx.wal.store_embeddings(ns, embed_turns_only(self.ctx.embed_docs, missing, s))
        self.read_diag["lazy_embedded_turns"] = len(missing)
        gated = gate_wal(qemb, delta, s["wal_sim_threshold"], s["b4_delta_top_m"])
        if self.needs_sentences:
            need = [t for t in gated if t.get("sent_embeddings") is None]
            if need:
                self.ctx.wal.store_embeddings(ns, embed_sentences_only(self.ctx.embed_docs, need, s,
                                                                       self.ctx.m["embed_dim"]))
            self.read_diag["lazy_embedded_sentence_turns"] = len(need)
        self.read_diag.update({
            "committed_epoch": committed, "n_uncommitted_turns": len(delta), "n_gated": len(gated),
            "gated": [(t["epoch"], t["turn"], round(t["sim"], 4)) for t in gated],
        })
        return facts, gated

    def _fact_line(self, f):
        return f"[OLDER MEMORY | epoch={f['source_epoch']} | {f['session_date']}] {f['entity_key']}: {f['fact_text']}"

    def _turn_line(self, t):
        return f"[NEWER MESSAGE | epoch={t['epoch']} | {t['date']}] {self.raw_turn(t['role'], t['content'])}"

    def _messages(self, prep, lines):
        user = "Memory:\n" + ("\n".join(lines) or "(none)") + f"\n\n{self.question_block(prep)}"
        return [{"role": "system", "content": READER_SYSTEM + "\n\n" + PRECEDENCE},
                {"role": "user", "content": user}]

    def _assemble(self, prep, facts, turns):
        items = [(f["source_epoch"], 0, self._fact_line(f)) for f in facts]
        items += [(t["epoch"], 1, self._turn_line(t)) for t in turns]
        return self._messages(prep, [x[2] for x in sorted(items, key=lambda x: (x[0], x[1]))])


class DeltaWALPrompt(_DeltaWALBase):
    name = "deltawal_prompt"

    def build_messages(self, prep, ns, ing, qemb):
        facts, gated = self._common(ns, qemb)
        extra = [{"epoch": t["epoch"], "turn": t["turn"], "role": t["role"], "sim": t["sim"]} for t in gated]
        return self._assemble(prep, facts, gated), [strip_emb(f) for f in facts], extra


class DeltaWALShadow(_DeltaWALBase):
    name = "deltawal_shadow"

    def build_messages(self, prep, ns, ing, qemb):
        facts, gated = self._common(ns, qemb)
        kept, sup = shadow_mask(facts, gated, self.ctx.cfg["systems"]["shadow_threshold"])
        self.read_diag["suppressed"] = [
            {"entity_key": f["entity_key"], "source_epoch": f["source_epoch"], "fact_text": f["fact_text"],
             "shadowed_by": f["shadowed_by"], "sim": round(f["shadow_sim"], 4)} for f in sup]
        extra = [{"epoch": t["epoch"], "turn": t["turn"], "role": t["role"], "sim": t["sim"]} for t in gated]
        return self._assemble(prep, kept, gated), [strip_emb(f) for f in kept], extra


class DeltaWALShadowV2(_DeltaWALBase):
    """Sentence-level shadow masking (proposed main method). Zero query-time LLM calls."""
    name = "deltawal_shadow_v2"
    needs_sentences = True

    def build_messages(self, prep, ns, ing, qemb):
        facts, gated = self._common(ns, qemb)
        kept, sup = shadow_mask_v2(facts, gated, self.ctx.cfg["systems"]["shadow_v2_threshold"])
        _, allsup = shadow_mask_v2(facts, gated, -1.0)      # raw max newer-sentence sim per fact
        self.read_diag["shadow_scores"] = [[f["entity_key"], f["source_epoch"], round(f["shadow_sim"], 4)]
                                           for f in allsup]
        self.read_diag["suppressed"] = [
            {"entity_key": f["entity_key"], "source_epoch": f["source_epoch"], "fact_text": f["fact_text"],
             "shadowed_by": f["shadowed_by"], "sim": round(f["shadow_sim"], 4)} for f in sup]
        extra = [{"epoch": t["epoch"], "turn": t["turn"], "role": t["role"], "sim": t["sim"]} for t in gated]
        return self._assemble(prep, kept, gated), [strip_emb(f) for f in kept], extra


class DeltaWALLazy(_DeltaWALBase):
    name = "deltawal_lazy"

    def build_messages(self, prep, ns, ing, qemb):
        facts, gated = self._common(ns, qemb)
        new_facts = []
        if gated:
            ex = self.ctx.extractor
            keys = sorted({f["entity_key"] for f in facts})
            lines = [f"[epoch {t['epoch']} | {t['date']}] [{t['role']}] {self.raw_text(t['role'], t['content'])}"
                     for t in gated]
            user = ("EXISTING KEYS:\n" + ("\n".join(keys) if keys else "(none)") + "\n\n"
                    "MESSAGES (each tagged with its epoch):\n" + "\n".join(lines) + "\n\n"
                    "For every fact also set `epoch` to the epoch of the message it came from.")
            res, _ = self.ctx.client.chat_json(ex.model, [
                {"role": "system", "content": SYSTEM_PROMPT.format(max_facts=ex.max_facts)},
                {"role": "user", "content": user}], schema=LAZY_SCHEMA, max_tokens=1200)
            self.read_llm_ms += 1000 * res.duration_s
            dates = {t["epoch"]: t["date"] for t in gated}
            max_epoch = max(dates)
            items = (res.parsed or {}).get("facts", []) if isinstance(res.parsed, dict) else []
            for it in items[: ex.max_facts]:
                if isinstance(it, dict) and str(it.get("fact", "")).strip():
                    ep = it.get("epoch")
                    ep = ep if isinstance(ep, int) and ep in dates else max_epoch
                    new_facts.append({"entity_key": normalize_key(str(it.get("entity_key", ""))),
                                      "fact_text": str(it["fact"]).strip(), "source_epoch": ep,
                                      "session_date": dates[ep]})
        newest = {}
        for f in new_facts:
            newest[f["entity_key"]] = max(newest.get(f["entity_key"], -1), f["source_epoch"])
        # epoch merge: a committed fact is replaced only by a strictly newer fact with the same key
        superseded = [f for f in facts if newest.get(f["entity_key"], -1) > f["source_epoch"]]
        kept = [f for f in facts if f not in superseded]
        self.read_diag["lazy_new_facts"] = [(f["entity_key"], f["source_epoch"], f["fact_text"]) for f in new_facts]
        self.read_diag["suppressed"] = [{"entity_key": f["entity_key"], "source_epoch": f["source_epoch"],
                                         "fact_text": f["fact_text"]} for f in superseded]
        items = [(f["source_epoch"], 0, self._fact_line(f)) for f in kept]
        items += [(f["source_epoch"], 1, f"[NEWER MESSAGE | epoch={f['source_epoch']} | {f['session_date']}] "
                                          f"{f['entity_key']}: {f['fact_text']}") for f in new_facts]
        msgs = self._messages(prep, [x[2] for x in sorted(items, key=lambda x: (x[0], x[1]))])
        return msgs, [strip_emb(f) for f in kept], [{"lazy_fact": f["entity_key"]} for f in new_facts]
