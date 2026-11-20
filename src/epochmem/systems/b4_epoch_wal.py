"""B4_epoch_wal (proposed): epoch-fenced write-ahead memory.

Write path: every turn is written through to a Redis WAL with a monotonic epoch *before*
the broker publish. The async worker commits facts with CAS fencing (an upsert only lands
if its source_epoch is newer than the stored one) and advances the contiguous
`committed_epoch` low-watermark.

Read path: vector facts from the committed index are merged with the WAL delta
(turn_epoch > committed_epoch, relevance-ranked, top-m). Every item carries its epoch and
commit status; the reader is told that higher epochs take precedence on conflicts and how
many sessions are still in flight.
"""
from __future__ import annotations

from .base import READER_SYSTEM, MemorySystem, cosine_top, trunc

PRECEDENCE = (
    "Memory items are tagged with an epoch: a monotonically increasing write sequence number "
    "(higher = more recent). COMMITTED items come from the indexed memory store; IN-FLIGHT items "
    "are raw conversation turns written after the store's committed watermark and not yet indexed. "
    "When items disagree about the same thing, trust the item with the higher epoch; treat lower-epoch "
    "values for that attribute as superseded."
)


class B4EpochWAL(MemorySystem):
    name = "B4_epoch_wal"
    write_mode = "async"
    policy = "cas"
    use_wal = True

    def build_messages(self, prep, ns, ing, qemb):
        s = self.ctx.cfg["systems"]
        facts = self.retrieve_facts(ns, qemb)
        committed = self.ctx.wal.committed(ns)
        delta = self.ctx.wal.delta(ns, committed)
        delta_texts = [trunc(d["content"], s["raw_turn_user_chars"] if d["role"] == "user"
                             else s["raw_turn_assistant_chars"]) for d in delta]
        top = cosine_top(qemb, self.ctx.embed_docs(delta_texts), s["b4_delta_top_m"]) if delta else []
        chosen = sorted((delta[i] for i in top), key=lambda d: (d["epoch"], d["turn"]))
        inflight_sessions = sorted({d["epoch"] for d in delta} - set(ing.applied_epochs))

        lines = [f"[epoch {f['source_epoch']} | {f['session_date']} | COMMITTED] {f['entity_key']}: {f['fact_text']}"
                 for f in sorted(facts, key=lambda f: f["source_epoch"])]
        lines += [f"[epoch {d['epoch']} | {d['date']} | IN-FLIGHT] {self.raw_turn(d['role'], d['content'])}"
                  for d in chosen]
        status = (f"Memory status: committed through epoch {committed}; "
                  f"{len(inflight_sessions)} newer session(s) not yet indexed"
                  + (f" (epochs {', '.join(map(str, inflight_sessions))})." if inflight_sessions else "."))
        user = f"{status}\n\nMemory:\n" + ("\n".join(lines) or "(none)") + f"\n\n{self.question_block(prep)}"
        msgs = [{"role": "system", "content": READER_SYSTEM + "\n\n" + PRECEDENCE},
                {"role": "user", "content": user}]
        extra = [{"epoch": d["epoch"], "turn": d["turn"], "role": d["role"]} for d in chosen]
        return msgs, facts, extra
