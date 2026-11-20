"""B3_recency: naive async + a raw recency buffer.

Same committed index as B2 (LWW, no fencing), plus the last K raw turns of the
conversation appended verbatim. No epoch tags, no precedence instructions, no write
fencing: tests whether a plain recency window resolves async staleness or merely creates
a context clash between stale indexed facts and fresh raw turns.
"""
from __future__ import annotations

from ..tokens import count_messages
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


class B3RecencyMatched(MemorySystem):
    """B3 with a token budget instead of a fixed K: appends as many trailing raw turns as fit
    so the whole reader prompt stays within `b3_matched_token_budget` (matched to DeltaWAL's
    mean prompt size). Same LWW index as B2; no watermark awareness, no relevance gating, no
    epoch tags, no precedence instructions."""
    name = "B3_recency_matched"
    write_mode = "async"
    policy = "lww"
    user_only = False

    def build_messages(self, prep, ns, ing, qemb):
        facts = self.retrieve_facts(ns, qemb)
        budget = self.ctx.cfg["systems"]["b3_matched_token_budget"]
        mem = "Retrieved memory facts:\n" + ("\n".join(map(self.fmt_fact_plain, facts)) or "(none)")
        all_turns = [t for ss in prep.stream for t in ss.session.turns if not self.user_only or t.role == "user"]

        def msgs_for(tail):
            recent = "Recent conversation:\n" + "\n".join(self.raw_turn(t.role, t.content) for t in tail)
            body = f"{mem}\n\n{recent}\n\n{self.question_block(prep)}" if tail else f"{mem}\n\n{self.question_block(prep)}"
            return [{"role": "system", "content": READER_SYSTEM}, {"role": "user", "content": body}]

        tail: list = []
        for t in reversed(all_turns):           # newest first; stop at the first turn that does not fit
            cand = [t] + tail
            if count_messages(msgs_for(cand)) > budget:
                break
            tail = cand
        self.read_diag["b3_tail_turns"] = len(tail)
        return msgs_for(tail), facts, [{"role": t.role, "content": t.content} for t in tail]


class B3RecencyMatchedUser(B3RecencyMatched):
    """Same token budget, trailing USER turns only (structural recency, not relevance gating)."""
    name = "B3_recency_matched_user"
    user_only = True
