"""DeltaWAL in ~150 lines: write-ahead log with epochs, atomic tail snapshot, relevance gate, epoch merge.

Mirrors src/epochmem/store/wal.py and src/epochmem/systems/deltawal.py, minus the broker, the extractor and
pgvector. Runs anywhere:

  uv run python examples/deltawal_minimal.py            # Redis at redis://localhost:6379/0 if reachable,
                                                        # otherwise an in-process stand-in with the same semantics
Embeddings default to a deterministic hashed bag of words, so no model is needed. Plug in a real embedder
(the paper uses qwen3-embedding:0.6b) through `embed=`; the threshold tau = 0.35 was calibrated for that model.
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

TAU, TOP_M, DIM = 0.35, 8, 256          # gate threshold, max gated turns, toy embedding size

# Write path: one atomic server-side call assigns the epoch and appends the session's raw turns, before the
# broker publish. (The repo does INCR + a MULTI pipeline; a single script makes the pair atomic by construction.)
APPEND_LUA = """
local epoch = redis.call('INCR', KEYS[1])
for i, turn in ipairs(ARGV) do
  local d = cjson.decode(turn)
  d.epoch = epoch; d.turn = i - 1
  redis.call('ZADD', KEYS[2], epoch, cjson.encode(d))
end
return epoch
"""
# Read path: watermark + every uncommitted turn in ONE round trip, so the tail can't change mid-read
# (same logic as _SNAPSHOT_LUA in src/epochmem/store/wal.py).
SNAPSHOT_LUA = """
local wm = tonumber(redis.call('GET', KEYS[1]) or '0')
local out = {}
for _, r in ipairs(redis.call('ZRANGEBYSCORE', KEYS[2], '(' .. wm, '+inf')) do
  if redis.call('SISMEMBER', KEYS[3], cjson.decode(r).epoch) == 0 then out[#out + 1] = r end
end
return {wm, out}
"""


def toy_embed(text: str) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    return v / (np.linalg.norm(v) + 1e-9)


class RedisWAL:
    def __init__(self, r, ns: str):
        self.r, self.k = r, {x: f"demo:{ns}:{x}" for x in ("epoch", "log", "applied", "committed")}
        r.delete(*self.k.values())
        self._append, self._snap = r.register_script(APPEND_LUA), r.register_script(SNAPSHOT_LUA)

    def append_session(self, date: str, turns: list[tuple[str, str]]) -> int:
        recs = [json.dumps({"role": ro, "content": c, "date": date}) for ro, c in turns]
        return int(self._append(keys=[self.k["epoch"], self.k["log"]], args=recs))

    def mark_committed(self, epoch: int) -> None:         # the async worker calls this after the store commit
        self.r.sadd(self.k["applied"], epoch)
        wm = int(self.r.get(self.k["committed"]) or 0)
        while self.r.sismember(self.k["applied"], wm + 1):
            wm += 1
        self.r.set(self.k["committed"], wm)

    def snapshot(self) -> tuple[int, list[dict]]:
        wm, recs = self._snap(keys=[self.k["committed"], self.k["log"], self.k["applied"]])
        return int(wm), sorted((json.loads(x) for x in recs), key=lambda d: (d["epoch"], d["turn"]))


class InProcessWAL:
    """Same semantics as RedisWAL for machines without Redis (single-threaded, so no extra locking needed)."""

    def __init__(self):
        self.epoch, self.log, self.applied, self.wm = 0, [], set(), 0

    def append_session(self, date, turns):
        self.epoch += 1
        self.log += [{"role": ro, "content": c, "date": date, "epoch": self.epoch, "turn": i}
                     for i, (ro, c) in enumerate(turns)]
        return self.epoch

    def mark_committed(self, epoch):
        self.applied.add(epoch)
        while self.wm + 1 in self.applied:
            self.wm += 1

    def snapshot(self):
        return self.wm, [d for d in self.log if d["epoch"] > self.wm and d["epoch"] not in self.applied]


def gate(query_vec, turns, embed, tau=TAU, top_m=TOP_M):
    """Keep uncommitted turns with cosine >= tau to the query (most similar first, at most top_m),
    returned in chronological order (as gate_wal in src/epochmem/systems/deltawal.py)."""
    if not turns:
        return []
    sims = np.stack([embed(t["content"]) for t in turns]) @ query_vec
    keep = [i for i in np.argsort(-sims, kind="stable") if sims[i] >= tau][:top_m]
    return sorted(({**turns[i], "sim": float(sims[i])} for i in keep), key=lambda t: (t["epoch"], t["turn"]))


def render(facts, gated) -> str:
    """Epoch-ordered merge of stored facts and gated WAL turns; facts sort before turns of the same epoch."""
    items = [(f["source_epoch"], 0, (f"[OLDER MEMORY | epoch={f['source_epoch']} | {f['date']}] "
                                     f"{f['entity_key']}: {f['fact']}")) for f in facts]
    items += [(t["epoch"], 1, f"[NEWER MESSAGE | epoch={t['epoch']} | {t['date']}] [{t['role']}] {t['content']}")
              for t in gated]
    return "\n".join(x[2] for x in sorted(items, key=lambda x: (x[0], x[1])))


def connect():
    try:
        import redis
    except ImportError:
        return InProcessWAL(), "in-process stand-in (redis-py not installed)"
    try:
        r = redis.Redis.from_url("redis://localhost:6379/0", decode_responses=True, socket_connect_timeout=0.5)
        r.ping()
        return RedisWAL(r, "user42"), "Redis (redis://localhost:6379/0)"
    except redis.exceptions.RedisError as e:              # unreachable
        return InProcessWAL(), f"in-process stand-in (Redis unavailable: {type(e).__name__})"


def main(embed=toy_embed) -> None:
    wal, backend = connect()
    print(f"backend: {backend}")
    e1 = wal.append_session("2026-03-02", [("user", "I just moved to Lisbon for a new job."), ("assistant", "Congrats!")])
    wal.mark_committed(e1)                                 # session 1 was extracted and committed ...
    facts = [{"entity_key": "user.home_city", "fact": "User lives in Lisbon.", "source_epoch": e1, "date": "2026-03-02"}]
    wal.append_session("2026-05-10", [("user", "Weekend plans: hiking, then a pottery class."), ("assistant", "Fun!")])
    wal.append_session("2026-06-21", [("user", "Thanks for the recipe!"), ("assistant", "Enjoy."),
                                      ("user", "Also, I relocated: I live in Porto now, closer to family."),
                                      ("assistant", "Porto is lovely.")])
    # ... sessions 2 and 3 are still in flight (lag 2): the fact store alone would answer "Lisbon".
    question = "Which city do I live in now?"
    wm, tail = wal.snapshot()
    gated = gate(embed(question), tail, embed)
    print(f"committed watermark: epoch {wm}; uncommitted turns: {len(tail)}; gated (cosine >= {TAU}): {len(gated)}")
    for t in gated:
        print(f"  epoch {t['epoch']} turn {t['turn']} sim={t['sim']:.2f}: {t['content']}")
    print("\nreader memory block:\n" + render(facts, gated))
    print("\nprecedence: the reader sees the Porto message at a higher epoch than the Lisbon fact.")


if __name__ == "__main__":
    main()
