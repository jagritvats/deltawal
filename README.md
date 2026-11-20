# EpochWAL-Mem: read-after-write staleness in asynchronous agent memory

Benchmark and reference implementation for measuring what happens when an agent's memory
pipeline ingests conversation turns asynchronously (Kafka/Redpanda -> extraction -> PostgreSQL/pgvector)
and a question arrives before the newest turns are indexed, or after commits were applied out of order.

Milestone 1 = local-only dry run (Ollama models, zero cloud spend) on a 20-question stratified
subset of LongMemEval-S (cleaned).

## Systems

| System | Write path | Store policy | Read path |
|---|---|---|---|
| `B1_sync` | extract + index synchronously before the query | in-order upsert | vector top-k facts |
| `B2_async` | Kafka async worker, last `lag` sessions in flight | naive last-writer-wins | vector top-k facts |
| `B3_recency` | as B2 | naive last-writer-wins | top-k facts + last K=8 raw turns (no epoch tags, no precedence rules) |
| `B4_epoch_wal` | Redis write-through WAL (monotonic epoch) before Kafka publish; worker advances `committed_epoch` | CAS fence: `ON CONFLICT ... WHERE EXCLUDED.source_epoch > facts.source_epoch` | top-k facts + relevance-ranked WAL delta (`epoch > committed_epoch`), epoch-tagged with newer-wins precedence and an in-flight status line |

## Fault axes (deterministic, no sleeps)

* `lag` in {0,1,3,5}: sessions still in flight when the query fires. The worker reads every
  message back from the broker by offset; a barrier lets exactly `n - lag` commits land.
* `p_ooo` in {0.0, 0.30}: straggler model; each message is delayed with probability `p_ooo` to a
  uniformly random later commit slot (simulated multi-worker reordering).
* `cutoff_mode`: `post_update` (default; the stream ends right after the last evidence session;
  abstention questions keep the full history) or `full_haystack`.

Real wall-clock latency is measured only in `scripts/bench_latency.py`.

## Metrics

* **Acc**: local LLM judge, LongMemEval per-type rubric.
* **CWR** (confident-wrong rate): wrong and did not abstain.
* **SRR-KU** (stale-read rate): knowledge-update questions where the answer is wrong *and* the judge
  confirms it matches the superseded pre-update value (taken from the earlier evidence session).
* **OOCR** (out-of-order corruption rate): share of updated `entity_key`s (written by >=2 committed
  epochs) whose stored `source_epoch` is below the newest committed epoch for that key. Fully
  deterministic; no LLM involved.
* **tokens**: mean reader prompt tokens (o200k_base tokenizer, model-agnostic).

## Engineering guarantees

* Every LLM/embedding call goes through a SQLite cache keyed by sha256(model + messages + params).
* Any non-localhost model endpoint raises unless `--allow-cloud` is passed (default off).
* Per-row append-only `runs/<run_id>/results.jsonl` with fsync; re-running a run id resumes.
* `seed=42` everywhere (sampling, commit schedules, decoding).
* Thinking is disabled (`think: false`) and extraction/judging use JSON-schema constrained output.
* Retrieval queries use the embedder's instruction prefix; indexed documents do not.

## Quickstart

```bash
uv sync
docker compose up -d --wait
uv run python scripts/download_data.py          # pinned revision, sha256-verified
uv run python scripts/verify_env.py             # endpoint, models, think-off, JSON, cache, infra, dataset
uv run pytest -q                                 # includes tests/test_cas_fencing.py
uv run python scripts/run_m1.py --run-id smoke3 --smoke 3
uv run python scripts/run_m1.py --run-id m1_dry20
uv run python scripts/summarize.py m1_dry20
uv run python scripts/bench_latency.py
```

## Layout

```
configs/m1_local.yaml      models, infra, grid, seeds
src/epochmem/
  guard.py cache.py llm.py tokens.py
  data/       base.py (MemoryDataset ABC) longmemeval.py sampler.py stream.py (cutoff modes)
  memory/     extract.py (entity_key, fact_text, source_epoch, event_date)
  store/      pg.py (LWW vs CAS upserts) wal.py (Redis WAL + watermark)
  stream/     bus.py (Kafka transport) schedule.py (p_ooo, lag barrier)
  systems/    b1_sync.py b2_async.py b3_recency.py b4_epoch_wal.py
  eval/       judge.py metrics.py
  runner.py
scripts/  verify_env.py download_data.py run_m1.py summarize.py bench_latency.py
tests/    test_core.py test_cas_fencing.py
```
