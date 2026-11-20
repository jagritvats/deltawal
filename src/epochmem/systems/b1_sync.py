"""B1_sync: synchronous extraction + vector indexing before the query (accuracy upper bound)."""
from __future__ import annotations

from .base import READER_SYSTEM, MemorySystem


class B1Sync(MemorySystem):
    name = "B1_sync"
    write_mode = "sync"
    policy = "lww"   # in-order writes, so LWW is already correct

    def build_messages(self, prep, ns, ing, qemb):
        facts = self.retrieve_facts(ns, qemb)
        mem = "Retrieved memory facts:\n" + ("\n".join(map(self.fmt_fact_plain, facts)) or "(none)")
        msgs = [{"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": f"{mem}\n\n{self.question_block(prep)}"}]
        return msgs, facts, []


_VSTORE: dict = {}


class B1SyncVersioned(MemorySystem):
    """Control: synchronous, NON-destructive, epoch-tagged. Every fact version is kept (append-only,
    keyed by (user_id, entity_key, source_epoch)); retrieved versions are rendered sorted by
    (source_epoch, event_date) with epoch + date tags and the same epoch-precedence instruction
    DeltaWAL uses. Isolates 'versioned storage + ordering' from 'async staleness handling'."""
    name = "B1_sync_versioned"
    write_mode = "sync"
    policy = "append"

    @property
    def store(self):
        key = id(self.ctx)
        if key not in _VSTORE:
            from ..store.pg import FactStore
            _VSTORE[key] = FactStore(self.ctx.cfg["infra"]["pg_dsn"], self.ctx.m["embed_dim"],
                                     table="facts_versioned", versioned=True)
        return _VSTORE[key]

    def build_messages(self, prep, ns, ing, qemb):
        from .deltawal import PRECEDENCE
        facts = self.retrieve_facts(ns, qemb)
        ordered = sorted(facts, key=lambda f: (f["source_epoch"], f.get("event_date") or ""))
        lines = [f"[MEMORY | epoch={f['source_epoch']} | {f['session_date']}] {f['entity_key']}: {f['fact_text']}"
                 for f in ordered]
        user = "Memory:\n" + ("\n".join(lines) or "(none)") + f"\n\n{self.question_block(prep)}"
        msgs = [{"role": "system", "content": READER_SYSTEM + "\n\n" + PRECEDENCE},
                {"role": "user", "content": user}]
        return msgs, facts, []
