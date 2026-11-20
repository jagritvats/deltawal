"""Redis write-through WAL with monotonic epochs and a committed low-watermark.

Layout per user namespace:
  wal:{ns}:epoch      INCR counter -> next epoch id (assigned before broker publish)
  wal:{ns}:log        ZSET score=epoch, member=JSON(turn record)
  wal:{ns}:applied    SET of epochs the async worker has committed to the vector store
  wal:{ns}:committed  contiguous low-watermark: every epoch <= it is committed
"""
from __future__ import annotations

import json

import redis

from ..data.stream import StreamSession


class RedisWAL:
    def __init__(self, url: str):
        self.r = redis.Redis.from_url(url, decode_responses=True)

    def ping(self) -> bool:
        return bool(self.r.ping())

    def reset(self, ns: str) -> None:
        self.r.delete(f"wal:{ns}:epoch", f"wal:{ns}:log", f"wal:{ns}:applied", f"wal:{ns}:committed")

    def next_epoch(self, ns: str) -> int:
        return int(self.r.incr(f"wal:{ns}:epoch"))

    def append_session(self, ns: str, epoch: int, ss: StreamSession) -> None:
        pipe = self.r.pipeline(transaction=True)
        for ti, t in enumerate(ss.session.turns):
            rec = {"epoch": epoch, "turn": ti, "role": t.role, "content": t.content,
                   "session_id": ss.session.session_id, "date": ss.session.date}
            pipe.zadd(f"wal:{ns}:log", {json.dumps(rec, ensure_ascii=False): epoch})
        pipe.execute()

    def mark_applied(self, ns: str, epoch: int) -> int:
        """Record a committed epoch and advance the contiguous low-watermark. Returns it."""
        self.r.sadd(f"wal:{ns}:applied", epoch)
        applied = {int(x) for x in self.r.smembers(f"wal:{ns}:applied")}
        wm = int(self.r.get(f"wal:{ns}:committed") or 0)
        while wm + 1 in applied:
            wm += 1
        self.r.set(f"wal:{ns}:committed", wm)
        return wm

    def committed(self, ns: str) -> int:
        return int(self.r.get(f"wal:{ns}:committed") or 0)

    def delta(self, ns: str, after_epoch: int) -> list[dict]:
        recs = [json.loads(m) for m in self.r.zrangebyscore(f"wal:{ns}:log", f"({after_epoch}", "+inf")]
        return sorted(recs, key=lambda r: (r["epoch"], r["turn"]))
