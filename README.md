# EpochWAL-Mem: read-after-write staleness in asynchronous agent memory

Benchmark and reference implementation for what happens when an agent's memory pipeline ingests
conversation turns asynchronously (Redpanda/Kafka -> fact extraction -> PostgreSQL/pgvector) and a
question arrives before the newest turns are indexed, or after commits were applied out of order.
It also measures what destructive state resolution (in-place overwrite on write, masking on read)
costs on questions about *earlier* values.

Evaluation: LongMemEval-S (cleaned) held-out split (138 questions), a 60-question non-update
regression split, Mem0 OSS 1.0.11 on the 60 held-out knowledge-update questions, and a 90-question
MemConflict split. One judge (qwen3.8-flash) throughout, validated against a larger judge
(Cohen's kappa in `paper/numbers_manifest.json`).

## Systems

| System | Write path | Store policy | Read path |
|---|---|---|---|
| `B1_sync` | extract + index synchronously before the query is answered | in-place upsert (mutable) | vector top-k facts |
| `B1_sync_versioned` | as B1 | append-only, one row per (entity_key, source_epoch) | top-k versions, epoch-tagged |
| `B2_async` | async worker; the last `lag` sessions are in flight | last-writer-wins | vector top-k facts |
| `B3_recency_matched_user` | as B2 | last-writer-wins | top-k facts + recent raw user turns, token budget matched to DeltaWAL |
| `deltawal_prompt` | Redis write-through WAL (monotonic epoch) before the broker publish | CAS fence on `source_epoch` | facts + relevance-gated WAL tail, epoch-ordered merge |
| `deltawal_shadow_v2` | as deltawal_prompt | CAS fence | as deltawal_prompt, but stored facts contradicted by a newer WAL sentence are masked |
| `deltawal_lazy` | as deltawal_prompt | CAS fence | in-flight sessions are extracted at query time |
| `deltawal_versioned` | as deltawal_prompt | append-only versions | versions + gated WAL tail, epoch-ordered merge (older values kept) |
| `Mem0_sync` / `Mem0_async` / `Mem0_deltawal` | Mem0 OSS 1.0.11 (`scripts/run_mem0.py`) | Mem0 ADD/UPDATE/DELETE | Mem0 search; `Mem0_deltawal` adds the WAL tail |

Frozen hyper-parameters (calibrated on a disjoint 20-question dev split, never tuned on test):
WAL relevance gate `wal_sim_threshold = 0.35`, Shadow v2 masking threshold `shadow_v2_threshold = 0.72`.

## Fault axes and metrics

* `lag` in {0, 1, 3, 5}: sessions still in flight when the query fires (deterministic barrier, no sleeps).
* `p_ooo` in {0.0, 0.3}: straggler model; commits are reordered with probability `p_ooo`.
* **Accuracy** per LongMemEval type rubric; **stale-read rate**: wrong and matching the superseded value;
  **OOCR**: updated keys left at an older epoch after reordering; **tokens**: mean reader prompt tokens.
* Wall-clock latency is measured only by `scripts/bench_latency.py` on the real stack.

## Repository layout

```
configs/            one YAML per experiment (frozen thresholds, model ids, budgets)
src/epochmem/       data loaders, WAL/stores, systems, judge, guarded + cached model clients
scripts/            runners, judges, reports, and build_paper_assets.py
tests/              unit tests (CAS fencing, WAL snapshot, DeltaWAL merge, Mem0 wrapper, paper assets)
paper/data/         snapshot.json.gz: slim per-row verdicts + benchmark summaries (no dataset text)
paper/tables/       generated LaTeX tables (booktabs)
paper/figures/      generated vector figures (+ preview/*.png at 200 dpi)
paper/numbers_manifest.json   every number in the tables/figures, with CIs and source hashes
```

## Reproducing the paper assets

All tables, figures and the manifest come from `scripts/build_paper_assets.py`; no number is typed by
hand. The build is deterministic (two builds are byte-identical) and fails on any matplotlib warning,
overlapping or clipped text, or a box label that does not fit its box.

### 1. From the committed snapshot (no data, models or infrastructure)

```bash
uv sync --group paper
bash reproduce_assets.sh                 # paper/tables/*.tex, paper/figures/*.pdf, numbers_manifest.json
bash reproduce_assets.sh --check-tex     # + compile everything with TeX Live (docker) at 6.5 in text width
uv run pytest -q                         # includes table <-> manifest consistency checks
uv run python scripts/audit_anonymity.py # double-blind + secret scan of tracked files, paper/ outputs, commit metadata
```

### 2. From cached run artifacts

With the run directories under `runs/` (and the SQLite model cache under `.cache/`), re-collect the
snapshot and rebuild:

```bash
bash reproduce_assets.sh --collect
```

Every report script below also re-runs from the cache at zero model cost.

### 3. From scratch

Prerequisites: Docker, [uv](https://docs.astral.sh/uv/), a local Ollama with `gemma4:12b-it-qat` and
`qwen3-embedding:0.6b`, and an OpenAI-compatible endpoint key in `.env` (`OPENCODE_API_KEY`) for the
cloud extractor/readers/judge. Every non-localhost call requires `--allow-cloud`; every call is cached
in SQLite, so re-running any command is free.

```bash
uv sync --group paper
docker compose up -d --wait                          # Redpanda, Redis, PostgreSQL/pgvector
uv run python scripts/download_data.py               # pinned revision, sha256-verified
uv run python scripts/verify_env.py
uv run pytest -q

# splits
uv run python scripts/make_heldout_slice.py          # 138 held-out questions (dev split excluded)
uv run python scripts/make_regression60_slice.py     # 60 non-update questions, disjoint
uv run python scripts/make_memconflict_split.py --write-split

# held-out LongMemEval-S (Table 1, 2, 6); extraction first (cloud extractor), then readers, then one judge pass
uv run python scripts/precompute_extractions.py --config configs/m2_heldout.yaml --workers 8 --allow-cloud
uv run python scripts/run_m1.py --config configs/m2_heldout.yaml --run-id m2ho_gemma --no-judge --core --cells "1:0,3:0,5:0,1:0.3,3:0.3" --allow-cloud
uv run python scripts/run_m1.py --config configs/m2_heldout.yaml --run-id m2ho_glm --reader opencode-zen/glm-5.3 --no-judge --core --cells "1:0,3:0,1:0.3" --allow-cloud
uv run python scripts/run_m1.py --config configs/m2_heldout.yaml --run-id m2ho_luna6 --reader opencode-zen/gpt-6-luna --no-judge --core --cells "1:0,3:0,1:0.3" --allow-cloud
for r in gemma:gemma4:12b-it-qat glm:opencode-zen/glm-5.3 luna6:opencode-zen/gpt-6-luna; do
  id=${r%%:*}; model=${r#*:}
  uv run python scripts/run_m1.py --config configs/m2_heldout.yaml --run-id m2ctl_$id --reader $model --no-judge \
      --systems B1_sync_versioned deltawal_prompt --core --cells "0:0" --allow-cloud
  uv run python scripts/run_m1.py --config configs/m2_heldout.yaml --run-id m2uni_$id --reader $model --no-judge \
      --systems deltawal_versioned --core --cells "0:0,1:0,3:0" --allow-cloud
done
uv run python scripts/run_m1.py --config configs/m2_heldout.yaml --run-id m2uni_gemma --no-judge \
    --systems deltawal_versioned --core --cells "5:0" --allow-cloud
uv run python scripts/judge_all.py --config configs/m2_heldout.yaml --runs m2ho_gemma m2ho_glm m2ho_luna6 \
    m2ctl_gemma m2ctl_glm m2ctl_luna6 m2uni_gemma m2uni_glm m2uni_luna6 --judge opencode-zen/qwen3.8-flash --allow-cloud
uv run python scripts/forensics_heldout.py --allow-cloud
uv run python scripts/sensitivity_shadow.py --run-id m2ho_gemma --judge opencode-zen/qwen3.8-flash --allow-cloud
uv run python scripts/judge_agreement.py --allow-cloud
uv run python scripts/bench_latency.py --config configs/m2_dev_local.yaml --questions 10 --no-reader
cp runs/bench_latency_m2.json runs/bench_latency_m2v2_no_reader.json

# non-update regression (Table 3)
uv run python scripts/precompute_extractions.py --config configs/m3_regression60.yaml --workers 16 --allow-cloud
for r in gemma:gemma4:12b-it-qat luna6:opencode-zen/gpt-6-luna; do
  id=${r%%:*}; model=${r#*:}
  uv run python scripts/run_m1.py --config configs/m3_regression60.yaml --run-id m3r60_$id --reader $model --no-judge --core \
      --cells "0:0,1:0" --systems B1_sync deltawal_versioned --allow-cloud
  uv run python scripts/run_m1.py --config configs/m3_regression60.yaml --run-id m3r60_$id --reader $model --no-judge --core \
      --cells "1:0" --systems B2_async B3_recency_matched_user --allow-cloud
done
uv run python scripts/run_m1.py --config configs/m3_regression60.yaml --run-id m3r60_glm --reader opencode-zen/glm-5.3 --no-judge \
    --core --cells "0:0,1:0" --systems B1_sync deltawal_versioned --allow-cloud
uv run python scripts/judge_all.py --config configs/m3_regression60.yaml --runs m3r60_gemma m3r60_luna6 m3r60_glm \
    --judge opencode-zen/qwen3.8-flash --audit-n 0 --summary-name judge_regression60_summary.json --allow-cloud

# Mem0 OSS 1.0.11 (Table 4): ingest once, then read with each reader, then the read-only history hook
uv run python scripts/run_mem0.py --run-id m3mem0_luna6 --reader opencode-zen/gpt-6-luna --ingest-only --ingest-workers 6 --allow-cloud
uv run python scripts/run_mem0.py --run-id m3mem0_luna6 --reader opencode-zen/gpt-6-luna --allow-cloud
uv run python scripts/run_mem0.py --run-id m3mem0_gemma --reader gemma4:12b-it-qat --allow-cloud
uv run python scripts/run_mem0.py --run-id m3mem0_glm --reader opencode-zen/glm-5.3 --systems Mem0_sync Mem0_deltawal --allow-cloud
for r in gemma:gemma4:12b-it-qat luna6:opencode-zen/gpt-6-luna glm:opencode-zen/glm-5.3; do
  uv run python scripts/run_mem0_history_hook.py --run-id m4mem0hist_${r%%:*} --reader ${r#*:} --allow-cloud
done
uv run python scripts/judge_all.py --config configs/m3_mem0.yaml --runs m3mem0_luna6 m3mem0_glm m3mem0_gemma \
    m4mem0hist_gemma m4mem0hist_luna6 m4mem0hist_glm --judge opencode-zen/qwen3.8-flash --audit-n 0 --allow-cloud

# MemConflict (Table 5)
uv run python scripts/precompute_extractions_cloudonly.py --config configs/m3_memconflict.yaml --workers 16 --allow-cloud
for r in gemma:gemma4:12b-it-qat luna6:opencode-zen/gpt-6-luna glmflash:opencode-zen/glm-5.3-flash; do
  id=${r%%:*}; model=${r#*:}
  uv run python scripts/run_m1.py --config configs/m3_memconflict.yaml --run-id m3mc_$id --reader $model --core --no-judge \
      --cells "0:0,1:0" --systems B1_sync deltawal_versioned --allow-cloud
  uv run python scripts/run_m1.py --config configs/m3_memconflict.yaml --run-id m3mc_$id --reader $model --core --no-judge \
      --cells "1:0" --systems B2_async B3_recency_matched_user --allow-cloud
done
uv run python scripts/judge_all.py --config configs/m3_memconflict.yaml --runs m3mc_gemma m3mc_luna6 m3mc_glmflash \
    --judge opencode-zen/qwen3.8-flash --audit-n 0 --summary-name judge_memconflict_summary.json --allow-cloud

# text reports and paper assets
uv run python scripts/report_unified.py
uv run python scripts/report_controls.py
uv run python scripts/report_m3.py --stage regression   # also: --stage mem0, --stage memconflict
bash reproduce_assets.sh --collect --check-tex
```

Never run two local Ollama jobs, or two writers to the same PostgreSQL tables, at the same time.
Cloud cost is dominated by fact extraction (one extractor call per session); `scripts/zen_spend.py` reports spend.

## Engineering guarantees

* Every LLM/embedding call goes through a SQLite cache keyed by sha256(model + messages + params).
* Any non-localhost model endpoint raises unless `--allow-cloud` is passed; per-process spend caps.
* Per-row append-only `runs/<run_id>/results.jsonl`; re-running a run id resumes.
* `seed = 42` everywhere (sampling, commit schedules, decoding, bootstrap).
* Bootstrap CIs: 2,000 resamples of questions; pooled multi-cell / multi-reader cells resample questions as clusters.
