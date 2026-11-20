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
