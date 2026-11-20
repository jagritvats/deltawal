"""Wall-clock latency microbenchmark (separate from accuracy runs, which never sleep).

Measures, over real infra with synthetic payloads and no LLM calls:
  write_ack      : client-visible write latency
                   B2 = broker publish ack; B4 = Redis WAL write-through + broker publish ack
  read_overhead  : query-time retrieval latency
                   B2 = pgvector top-k; B4 = pgvector top-k + Redis watermark read + delta scan
  visibility_lag : time from write ack until the fact is readable by the B2 retriever
                   (worker consume + upsert), vs. 0 for B4 (delta visible at ack)
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time

from epochmem.config import load_config
from epochmem.data.base import Session, Turn
from epochmem.data.stream import StreamSession
from epochmem.memory.extract import Fact
from epochmem.store.pg import FactStore
from epochmem.store.wal import RedisWAL
from epochmem.stream.bus import SessionBus


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summary(xs):
    return {"p50_ms": round(1000 * pct(xs, 50), 3), "p99_ms": round(1000 * pct(xs, 99), 3),
            "mean_ms": round(1000 * statistics.mean(xs), 3), "n": len(xs)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_local.yaml")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config(args.config)
    infra, dim = cfg["infra"], cfg["models"]["embed_dim"]
    rng = random.Random(cfg["seed"])
    store, wal = FactStore(infra["pg_dsn"], dim), RedisWAL(infra["redis_url"])
    bus = SessionBus(infra["kafka_bootstrap"], infra["kafka_topic"] + ".bench")
    bus.ensure_topic()
    vec = lambda: [rng.uniform(-1, 1) for _ in range(dim)]
    ns = f"bench-{int(time.time())}"
    store.reset(ns)
    wal.reset(ns)
    for i in range(500):  # background index
        store.upsert(ns, Fact(f"user.k{i}", f"fact {i}", 0, "unknown", "d"), vec(), "lww")

    w_b2, w_b4, r_b2, r_b4, vis = [], [], [], [], []
    for i in range(args.n):
        ss = StreamSession(i + 1, Session(f"s{i}", "d", tuple(Turn("user", "x" * 400) for _ in range(10))), False)
        msg = {"ns": ns, "epoch": i + 1, "turns": [{"role": "user", "content": "x" * 400}] * 10}
        t0 = time.perf_counter()
        bus.publish(ns, [msg])
        w_b2.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        e = wal.next_epoch(ns)
        wal.append_session(ns, e, ss)
        bus.publish(ns, [msg])
        t_ack = time.perf_counter()
        w_b4.append(t_ack - t0)

        # visibility via the async path: worker consumes then upserts
        offs = bus.publish(ns, [msg])
        t_pub = time.perf_counter()
        bus.fetch(ns, offs)
        store.upsert(ns, Fact(f"user.k{i % 50}", f"v{i}", i + 1, "unknown", "d"), vec(), "cas")
        vis.append(time.perf_counter() - t_pub)
        if i % 3 == 0:
            wal.mark_applied(ns, e)

        q = vec()
        t0 = time.perf_counter()
        store.search(ns, q, 12)
        r_b2.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        store.search(ns, q, 12)
        wal.delta(ns, wal.committed(ns))
        r_b4.append(time.perf_counter() - t0)

    out = {"write_ack": {"B2_async": summary(w_b2), "B4_epoch_wal": summary(w_b4)},
           "read_overhead": {"B2_async": summary(r_b2), "B4_epoch_wal": summary(r_b4)},
           "visibility_lag_async_path": summary(vis)}
    print(json.dumps(out, indent=2))
    store.reset(ns)
    wal.reset(ns)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
