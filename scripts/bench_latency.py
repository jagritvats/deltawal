"""Wall-clock latency benchmark on real infra and real dev-split sessions.

Separate from accuracy runs (which never sleep and never time anything that matters).
For each dev question the stream is published to Redpanda while a real background worker
thread consumes it, embeds facts, CAS-upserts into PostgreSQL, advances committed_epoch in
Redis and trims the WAL. The worker stops after n-lag commits (lag=3 by default) so the read
paths see a genuine in-flight tail.

Write-ack (client-visible, per session):
  B1_sync   : extraction (recorded live wall time of the original call) + live fact embedding + upserts
  B2_async  : broker publish ack
  DeltaWAL  : Redis WAL write-through of raw turns + epoch + broker publish ack (turn/sentence
              embeddings are computed by a background embedder thread; reads lazily embed stragglers)
Read path (per question, live embedding/LLM calls, no cache):
  query embedding + retrieval/merge, excluding the final reader decode (deltawal_lazy includes
  its live query-time extraction call); end_to_end additionally includes one live reader call.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time

from confluent_kafka import Consumer, TopicPartition

from epochmem.config import load_config, repo_path
from epochmem.guard import set_allow_cloud
from epochmem.llm import OllamaClient
from epochmem.memory.wal_embed import BackgroundEmbedder
from epochmem.runner import build_ctx, select_instances
from epochmem.systems import REGISTRY
from epochmem.systems.base import Ctx, session_message

READ_SYSTEMS = ["B1_sync", "B2_async", "deltawal_prompt", "deltawal_shadow_v2", "deltawal_lazy"]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summ(xs_s):
    ms = [1000 * x for x in xs_s]
    return {"p50_ms": round(pct(ms, 50), 2), "p95_ms": round(pct(ms, 95), 2), "p99_ms": round(pct(ms, 99), 2),
            "mean_ms": round(statistics.mean(ms), 2) if ms else None, "n": len(ms)}


class Worker(threading.Thread):
    """Consumes this namespace's session messages from Redpanda and commits them (CAS) until
    `limit` commits (the lag barrier)."""

    def __init__(self, ctx: Ctx, live: OllamaClient, prep, ns: str, facts_by_epoch, limit: int, t_pub: dict):
        super().__init__(daemon=True)
        self.ctx, self.live, self.prep, self.ns = ctx, live, prep, ns
        self.facts, self.limit, self.t_pub = facts_by_epoch, limit, t_pub
        self.visible_s: list[float] = []
        self.done = threading.Event()
        self.ready = threading.Event()
        self.consumer = Consumer({"bootstrap.servers": ctx.bus.bootstrap, "broker.address.family": "v4",
                                  "group.id": f"bench-worker-{ns}", "enable.auto.commit": False,
                                  "fetch.wait.max.ms": 5, "fetch.min.bytes": 1})
        lo, hi = self.consumer.get_watermark_offsets(TopicPartition(ctx.bus.topic, 0), timeout=10)
        self.consumer.assign([TopicPartition(ctx.bus.topic, 0, hi)])   # only messages published from now on

    def run(self):
        applied = 0
        while applied < self.limit:
            m = self.consumer.poll(0.05)
            if m is None or m.error() or m.key() != self.ns.encode():
                continue
            msg = json.loads(m.value())
            e = msg["epoch"]
            fs = self.facts.get(e, [])
            embs = self.live.embed(self.ctx.m["embedder"], [f.doc_text() for f in fs]) if fs else []
            for f, v in zip(fs, embs):
                self.ctx.store.upsert(self.ns, f, v, "cas")
            self.ctx.wal.mark_applied(self.ns, e)          # advances committed_epoch + trims WAL
            self.visible_s.append(time.perf_counter() - self.t_pub[e])
            applied += 1
        self.consumer.close()
        self.done.set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_dev_local.yaml")
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--lag", type=int, default=3)
    ap.add_argument("--no-reader", action="store_true", help="skip live end-to-end reader calls")
    args = ap.parse_args()
    set_allow_cloud(False)
    cfg = load_config(args.config)
    ctx, _ = build_ctx(cfg)                       # cached client: only used to fetch cached extractions
    m = cfg["models"]
    live = OllamaClient(m["endpoint"], None, num_ctx=m["num_ctx"], seed=cfg["seed"], think=m["think"])
    lctx = Ctx(cfg=cfg, client=live, extractor=ctx.extractor, store=ctx.store, wal=ctx.wal, bus=ctx.bus, seed=ctx.seed)
    lctx.wal.trim = True
    preps = [p for p in select_instances(cfg, None)][: args.questions]

    w_b1, w_b2, w_dw, vis, vis_with_extract = [], [], [], [], []
    read = {s: [] for s in READ_SYSTEMS}
    read_noemb = {s: [] for s in READ_SYSTEMS}
    qemb_s = []
    e2e = {s: [] for s in READ_SYSTEMS}
    wal_sizes = []
    bg_done, lazy_fill = [], {s: [] for s in READ_SYSTEMS}
    cold = {"deltawal_prompt": [], "deltawal_shadow_v2": []}
    cold_n = {"deltawal_prompt": [], "deltawal_shadow_v2": []}
    stamp = int(time.time())
    for prep in preps:
        facts = ctx.extractor.extract_instance(prep)
        ns_sync, ns = f"bench{stamp}:sync:{prep.inst.qid}", f"bench{stamp}:dw:{prep.inst.qid}"
        for n_ in (ns_sync, ns):
            ctx.store.reset(n_)
        ctx.wal.reset(ns)
        t_pub: dict[int, float] = {}
        worker = Worker(ctx, live, prep, ns, facts, prep.n - args.lag, t_pub)
        worker.start()
        bg = BackgroundEmbedder(ctx.wal, ns, lambda ts: live.embed(m["embedder"], ts), cfg["systems"], m["embed_dim"])
        bg.start()
        for ss in prep.stream:
            # B1_sync write path (extraction latency = recorded live wall of the original call)
            t0 = time.perf_counter()
            fs = facts.get(ss.epoch, [])
            embs = live.embed(m["embedder"], [f.doc_text() for f in fs]) if fs else []
            for f, v in zip(fs, embs):
                ctx.store.upsert(ns_sync, f, v, "lww")
            w_b1.append(time.perf_counter() - t0 + ctx.extractor.durations.get((prep.inst.qid, ss.epoch), 0.0))
            # B2_async write path: publish ack only (separate throwaway key, not consumed)
            msg = session_message(ss, ns)
            t0 = time.perf_counter()
            ctx.bus.publish(ns + ":b2", [msg])
            w_b2.append(time.perf_counter() - t0)
            # DeltaWAL write path: raw WAL write-through + publish ack (embedding is off the ack path)
            t0 = time.perf_counter()
            e = ctx.wal.next_epoch(ns)
            ctx.wal.append_session(ns, e, ss)
            t_pub[e] = time.perf_counter()             # visibility is measured from publish start
            ctx.bus.publish(ns, [msg])
            w_dw.append(time.perf_counter() - t0)
        worker.done.wait(timeout=600)
        t_drain = time.perf_counter()
        while ctx.wal.r.llen(f"wal:{ns}:pending") and time.perf_counter() - t_drain < 120:
            time.sleep(0.05)                       # benchmark only: wait for background embedding (warm reads)
        time.sleep(0.5)
        bg.stop()
        bg.join(timeout=30)
        bg_done.append(bg.embedded_sessions)
        vis.extend(worker.visible_s)
        vis_with_extract.extend(v + ctx.extractor.durations.get((prep.inst.qid, i + 1), 0.0)
                                for i, v in enumerate(worker.visible_s))
        wal_sizes.append({"qid": prep.inst.qid, "n_sessions": prep.n, "committed_epoch": ctx.wal.committed(ns),
                          "wal_log_turns_after_trim": ctx.wal.log_size(ns)})
        # read paths
        for s in READ_SYSTEMS:
            sysobj = REGISTRY[s](lctx)
            sysobj.read_diag, sysobj.read_llm_ms = {}, 0.0
            target = ns_sync if s == "B1_sync" else ns
            t0 = time.perf_counter()
            qemb = lctx.embed_query(prep.inst.question)
            t_emb = time.perf_counter()
            msgs, _, _ = sysobj.build_messages(prep, target, None, qemb)
            t_read = time.perf_counter() - t0
            read[s].append(t_read)
            read_noemb[s].append(time.perf_counter() - t_emb)
            qemb_s.append(t_emb - t0)
            lazy_fill[s].append(sysobj.read_diag.get("lazy_embedded_turns", 0))
            if not args.no_reader:
                t1 = time.perf_counter()
                live.chat(m["reader"], msgs, max_tokens=256)
                e2e[s].append(t_read + time.perf_counter() - t1)
        # cold read: the query arrives before any background embedding of the uncommitted tail
        for s in ("deltawal_prompt", "deltawal_shadow_v2"):
            _, unc = ctx.wal.snapshot(ns)
            fields = [f"{t['epoch']}:{t['turn']}" for t in unc]
            if fields:
                ctx.wal.r.hdel(f"wal:{ns}:emb", *fields)
                ctx.wal.r.hdel(f"wal:{ns}:semb", *fields)
            sysobj = REGISTRY[s](lctx)
            sysobj.read_diag, sysobj.read_llm_ms = {}, 0.0
            t0 = time.perf_counter()
            qemb = lctx.embed_query(prep.inst.question)
            t_emb = time.perf_counter()
            sysobj.build_messages(prep, ns, None, qemb)
            cold[s].append(time.perf_counter() - t_emb)
            cold_n[s].append(sysobj.read_diag.get("lazy_embedded_turns", 0))
        ctx.store.reset(ns_sync)
        ctx.store.reset(ns)
        ctx.wal.reset(ns)
        print(f"benched {prep.inst.qid}", flush=True)

    out = {
        "config": {"questions": len(preps), "lag": args.lag, "embedder": m["embedder"], "reader": m["reader"]},
        "write_ack": {"B1_sync": summ(w_b1), "B2_async": summ(w_b2), "deltawal_*": summ(w_dw)},
        "async_visibility_publish_start_to_committed_excl_extraction": summ(vis),
        "async_visibility_incl_recorded_extraction": summ(vis_with_extract),
        "read_path_excl_reader": {s: summ(v) for s, v in read.items()},
        "read_path_excl_reader_and_query_embedding": {s: summ(v) for s, v in read_noemb.items()},
        "query_embedding": summ(qemb_s),
        "end_to_end_incl_live_reader": {s: summ(v) for s, v in e2e.items()} if not args.no_reader else None,
        "wal_trim_check": wal_sizes,
        "background_embedded_sessions_per_question": bg_done,
        "cold_read_excl_reader_and_query_embedding": {s: summ(v) for s, v in cold.items()},
        "cold_read_lazy_embedded_turns_mean": {s: (sum(v) / len(v) if v else 0) for s, v in cold_n.items()},
        "read_lazy_embedded_turns_mean": {s: (sum(v) / len(v) if v else 0) for s, v in lazy_fill.items()},
    }
    path = repo_path("runs/bench_latency_m2.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "wal_trim_check"}, indent=1))
    print("wal trim:", wal_sizes[:3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
