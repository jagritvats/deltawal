"""B3_recency: naive async + a raw recency buffer.

Same committed index as B2 (LWW, no fencing), plus the last K raw turns of the
conversation appended verbatim. No epoch tags, no precedence instructions, no write
fencing: tests whether a plain recency window resolves async staleness or merely creates
a context clash between stale indexed facts and fresh raw turns.
"""
from __future__ import annotations

from .base import READER_SYSTEM, MemorySystem


class B3Recency(MemorySystem):
    name = "B3_recency"
    write_mode = "async"
    policy = "lww"

    def build_messages(self, prep, ns, ing, qemb):
        facts = self.retrieve_facts(ns, qemb)
        k = self.ctx.cfg["systems"]["b3_tail_turns"]
        tail = [t for ss in prep.stream for t in ss.session.turns][-k:]
        extra = [{"role": t.role, "content": t.content} for t in tail]
        mem = "Retrieved memory facts:\n" + ("\n".join(map(self.fmt_fact_plain, facts)) or "(none)")
        recent = "Recent conversation:\n" + "\n".join(self.raw_turn(t.role, t.content) for t in tail)
        msgs = [{"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": f"{mem}\n\n{recent}\n\n{self.question_block(prep)}"}]
        return msgs, facts, extra
