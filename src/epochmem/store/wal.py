"""Redis write-through WAL with monotonic epochs and a committed low-watermark.

Layout per user namespace:
  wal:{ns}:epoch      INCR counter -> next epoch id (assigned before broker publish)
  wal:{ns}:log        ZSET score=epoch, member=JSON(turn record)
  wal:{ns}:emb        HASH "epoch:turn" -> float32 bytes (turn embedding, computed at write time
                      so the read path needs no model call to gate/shadow WAL turns)
  wal:{ns}:applied    SET of epochs the async worker has committed to the vector store
  wal:{ns}:committed  contiguous low-watermark: every epoch <= it is committed
"""
from __future__ import annotations

import json

import numpy as np
import redis

from ..data.stream import StreamSession


class RedisWAL:
    def __init__(self, url: str, trim: bool = False, dim: int = 1024):
        self.dim = dim
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.rb = redis.Redis.from_url(url, decode_responses=False)
        self.trim = trim

    def ping(self) -> bool:
        return bool(self.r.ping())

    def reset(self, ns: str) -> None:
        self.r.delete(*(f"wal:{ns}:{k}" for k in ("epoch", "log", "emb", "semb", "applied", "committed", "pending")))

    def next_epoch(self, ns: str) -> int:
        return int(self.r.incr(f"wal:{ns}:epoch"))

    def append_session(self, ns: str, epoch: int, ss: StreamSession,
                       embeddings: list[list[float]] | None = None) -> None:
        """Synchronous write-ack path: raw turns + epoch (one pipelined round trip). Embeddings are
        normally produced later by the background embedder (`pending` queue)."""
        pipe = self.rb.pipeline(transaction=True)
        for ti, t in enumerate(ss.session.turns):
            rec = {"epoch": epoch, "turn": ti, "role": t.role, "content": t.content,
                   "session_id": ss.session.session_id, "date": ss.session.date}
            pipe.zadd(f"wal:{ns}:log", {json.dumps(rec, ensure_ascii=False).encode(): epoch})
            if embeddings is not None:
                pipe.hset(f"wal:{ns}:emb", f"{epoch}:{ti}", np.asarray(embeddings[ti], dtype=np.float32).tobytes())
        if embeddings is None:
            pipe.rpush(f"wal:{ns}:pending", epoch)
        pipe.execute()

    def store_embeddings(self, ns: str, items: list[tuple[int, int, list[float], np.ndarray]]) -> None:
        """items: (epoch, turn, turn_embedding, sentence_matrix[k, d])."""
        if not items:
            return
        pipe = self.rb.pipeline(transaction=False)
        for e, t, temb, smat in items:          # either part may be None (partial lazy fill)
            if temb is not None:
                pipe.hset(f"wal:{ns}:emb", f"{e}:{t}", np.asarray(temb, dtype=np.float32).tobytes())
            if smat is not None:
                pipe.hset(f"wal:{ns}:semb", f"{e}:{t}", np.asarray(smat, dtype=np.float32).tobytes())
        pipe.execute()

    def pop_pending(self, ns: str, timeout: float = 0.0) -> int | None:
        if timeout > 0:
            r = self.r.blpop([f"wal:{ns}:pending"], timeout=timeout)
            return int(r[1]) if r else None
        v = self.r.lpop(f"wal:{ns}:pending")
        return int(v) if v is not None else None

    def session_turns(self, ns: str, epoch: int) -> list[dict]:
        return sorted((json.loads(m) for m in self.r.zrangebyscore(f"wal:{ns}:log", epoch, epoch)),
                      key=lambda d: d["turn"])

    def mark_applied(self, ns: str, epoch: int) -> int:
        """Record a committed epoch, advance the contiguous low-watermark, optionally trim. Returns it."""
        self.r.sadd(f"wal:{ns}:applied", epoch)
        wm = int(self.r.get(f"wal:{ns}:committed") or 0)
        new = wm
        while self.r.sismember(f"wal:{ns}:applied", new + 1):
            new += 1
        if new != wm:
            self.r.set(f"wal:{ns}:committed", new)
            if self.trim:
                self._trim(ns, wm, new)
        return new

    def _trim(self, ns: str, old_wm: int, new_wm: int) -> None:
        recs = self.r.zrangebyscore(f"wal:{ns}:log", f"({old_wm}", new_wm)
        pipe = self.r.pipeline(transaction=True)
        pipe.zremrangebyscore(f"wal:{ns}:log", "-inf", new_wm)
        fields = [f"{json.loads(m)['epoch']}:{json.loads(m)['turn']}" for m in recs]
        if fields:
            pipe.hdel(f"wal:{ns}:emb", *fields)
            pipe.hdel(f"wal:{ns}:semb", *fields)
        pipe.execute()

    def committed(self, ns: str) -> int:
        return int(self.r.get(f"wal:{ns}:committed") or 0)

    def applied(self, ns: str) -> set[int]:
        return {int(x) for x in self.r.smembers(f"wal:{ns}:applied")}

    def log_size(self, ns: str) -> int:
        return int(self.r.zcard(f"wal:{ns}:log"))

    # One round trip: watermark + uncommitted turns + their embeddings, atomically on the server.
    _SNAPSHOT_LUA = """
    local wm = tonumber(redis.call('GET', KEYS[1]) or '0')
    local recs = redis.call('ZRANGEBYSCORE', KEYS[2], '(' .. wm, '+inf')
    local out_r, out_e, out_s = {}, {}, {}
    for _, r in ipairs(recs) do
        local d = cjson.decode(r)
        if redis.call('SISMEMBER', KEYS[3], d.epoch) == 0 then
            local f = d.epoch .. ':' .. d.turn
            out_r[#out_r + 1] = r
            out_e[#out_e + 1] = redis.call('HGET', KEYS[4], f) or ''
            out_s[#out_s + 1] = redis.call('HGET', KEYS[5], f) or ''
        end
    end
    return {wm, out_r, out_e, out_s}
    """

    def snapshot(self, ns: str) -> tuple[int, list[dict]]:
        """(committed_epoch, uncommitted turns with embeddings) in a single server-side call.
        Equivalent to committed() + delta(ns, committed, uncommitted_only=True, with_embeddings=True)."""
        if not hasattr(self, "_snap"):
            self._snap = self.rb.register_script(self._SNAPSHOT_LUA)
        wm, recs, embs, sembs = self._snap(keys=[f"wal:{ns}:committed", f"wal:{ns}:log", f"wal:{ns}:applied",
                                                 f"wal:{ns}:emb", f"wal:{ns}:semb"])
        out = []
        for r, b, sb in zip(recs, embs, sembs):
            d = json.loads(r)
            d["embedding"] = np.frombuffer(b, dtype=np.float32) if b else None
            d["sent_embeddings"] = np.frombuffer(sb, dtype=np.float32).reshape(-1, self.dim) if sb else None
            out.append(d)
        out.sort(key=lambda r: (r["epoch"], r["turn"]))
        return int(wm), out

    def delta(self, ns: str, after_epoch: int, uncommitted_only: bool = False,
              with_embeddings: bool = False) -> list[dict]:
        recs = [json.loads(m) for m in self.r.zrangebyscore(f"wal:{ns}:log", f"({after_epoch}", "+inf")]
        if uncommitted_only:
            done = self.applied(ns)
            recs = [r for r in recs if r["epoch"] not in done]
        recs.sort(key=lambda r: (r["epoch"], r["turn"]))
        if with_embeddings and recs:
            raw = self.rb.hmget(f"wal:{ns}:emb", [f"{r['epoch']}:{r['turn']}" for r in recs])
            for r, b in zip(recs, raw):
                r["embedding"] = np.frombuffer(b, dtype=np.float32) if b is not None else None
        return recs
