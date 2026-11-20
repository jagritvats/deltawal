"""Verify the local model endpoint, cache, guard, infra and dataset before any run."""
from __future__ import annotations

import argparse
import sys
import time

from epochmem.cache import DiskCache, cache_key
from epochmem.config import load_config, repo_path
from epochmem.data.longmemeval import LongMemEvalDataset
from epochmem.guard import CloudCallBlocked, check_url, set_allow_cloud
from epochmem.llm import OllamaClient
from epochmem.memory.extract import FACT_SCHEMA

OK, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, OK if ok else FAIL, detail))
    print(f"[{OK if ok else FAIL}] {name} {detail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_local.yaml")
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    m = cfg["models"]

    # 1. guard
    try:
        check_url("https://api.example.com/v1/chat/completions")
        record("guard blocks non-localhost", args.allow_cloud, "(no error raised)")
    except CloudCallBlocked:
        record("guard blocks non-localhost", not args.allow_cloud)

    # 2. endpoint + models
    cache = DiskCache(repo_path(".cache/verify_cache.sqlite"))
    client = OllamaClient(m["endpoint"], cache, num_ctx=m["num_ctx"], seed=cfg["seed"], think=m["think"])
    record("ollama reachable", True, f"version={client.version()}")
    tags = set(client.list_models())
    for role in ("reader", "extractor", "judge", "embedder"):
        record(f"model present: {role}", m[role] in tags, m[role])

    # 3. thinking disabled + JSON-schema output
    msgs = [{"role": "user", "content": "Session: I just adopted a beagle named Pixel on 2023-04-02. "
                                        "Extract facts as JSON with entity_key, fact, event_date."}]
    t0 = time.perf_counter()
    r = client.chat(m["extractor"], msgs, schema=FACT_SCHEMA, max_tokens=300)
    dt = time.perf_counter() - t0
    record("think disabled (no thinking trace)", r.thinking == "", f"thinking_chars={len(r.thinking)}")
    record("JSON-schema output parses", r.parse_error is None and "facts" in (r.parsed or {}),
           f"{dt:.1f}s facts={r.parsed.get('facts') if isinstance(r.parsed, dict) else r.content[:80]}")

    # 4. cache determinism: identical request must be served from cache with identical content
    r2 = client.chat(m["extractor"], msgs, schema=FACT_SCHEMA, max_tokens=300)
    record("cache hit on repeat call", r2.cached and r2.content == r.content)
    k1 = cache_key({"a": 1, "b": [1, 2]})
    k2 = cache_key({"b": [1, 2], "a": 1})
    record("cache key order-invariant", k1 == k2)

    # 5. embeddings
    v = client.embed(m["embedder"], [m["query_instruction"] + "What is my dog's name?", "user.dog.name: Pixel"])
    record("embedding dim", len(v[0]) == m["embed_dim"] and len(v[1]) == m["embed_dim"], f"dim={len(v[0])}")

    # 6. infra
    try:
        import psycopg
        with psycopg.connect(cfg["infra"]["pg_dsn"], connect_timeout=5) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
            ver = c.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
        record("postgres + pgvector", True, f"pgvector={ver}")
    except Exception as e:
        record("postgres + pgvector", False, repr(e)[:200])
    try:
        from epochmem.store.wal import RedisWAL
        record("redis", RedisWAL(cfg["infra"]["redis_url"]).ping())
    except Exception as e:
        record("redis", False, repr(e)[:200])
    try:
        from epochmem.stream.bus import SessionBus
        bus = SessionBus(cfg["infra"]["kafka_bootstrap"], cfg["infra"]["kafka_topic"])
        bus.ensure_topic()
        offs = bus.publish("verify", [{"epoch": 1}, {"epoch": 2}])
        back = bus.fetch("verify", offs)
        record("kafka publish/fetch roundtrip", [b["epoch"] for b in back] == [1, 2], f"offsets={offs}")
    except Exception as e:
        record("kafka publish/fetch roundtrip", False, repr(e)[:200])

    # 7. dataset
    d = cfg["dataset"]
    path = repo_path(d["path"])
    try:
        ds = LongMemEvalDataset(path, expected_sha256=d["sha256"])
        digest = ds.verify()
        size_ok = path.stat().st_size == d["size_bytes"]
        insts = ds.load()
        record("dataset sha256 + size", size_ok, digest[:16])
        record("dataset parses (500 q)", len(insts) == 500)
    except Exception as e:
        record("dataset", False, repr(e)[:200])
    try:
        LongMemEvalDataset(path.with_name("longmemeval_oracle.json"))
        record("oracle split refused", False)
    except ValueError:
        record("oracle split refused", True)

    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
