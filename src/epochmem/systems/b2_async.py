"""B2_async: Kafka async ingestion; reads only what the worker has committed.

Upserts are naive last-writer-wins by arrival order, so reordered commits can overwrite a
newer value with an older one.
"""
from __future__ import annotations

from .base import READER_SYSTEM, MemorySystem


class B2Async(MemorySystem):
    name = "B2_async"
    write_mode = "async"
    policy = "lww"

    def build_messages(self, prep, ns, ing, qemb):
        facts = self.retrieve_facts(ns, qemb)
        mem = "Retrieved memory facts:\n" + ("\n".join(map(self.fmt_fact_plain, facts)) or "(none)")
        msgs = [{"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": f"{mem}\n\n{self.question_block(prep)}"}]
        return msgs, facts, []
