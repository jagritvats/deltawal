"""Resumable experiment runner: one append-only JSONL row per (instance, system, lag, p_ooo)."""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from .cache import DiskCache
from .config import Config, repo_path
from .data.longmemeval import LongMemEvalDataset
from .data.sampler import smoke_subset, stratified_sample
from .data.stream import PreparedInstance, prepare
from .eval.judge import Judge
from .llm import OllamaClient
from .memory.extract import FactExtractor
from .store.pg import FactStore
from .store.wal import RedisWAL
from .stream.bus import SessionBus
from .systems import REGISTRY
from .systems.base import Ctx


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def done_keys(self) -> set[str]:
        if not self.path.exists():
            return set()
        keys = set()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    keys.add(json.loads(line)["row_key"])
                except (json.JSONDecodeError, KeyError):
                    continue  # tolerate a torn final line after a crash
        return keys

    def append(self, row: dict, path: Path | None = None) -> None:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock, (path or self.path).open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())

    def append_side(self, name: str, row: dict) -> None:
        self.append(row, self.path.parent / name)


def build_ctx(cfg: Config) -> tuple[Ctx, Judge]:
    m, infra, mem = cfg["models"], cfg["infra"], cfg["memory"]
    cache = DiskCache(repo_path(cfg["runtime"]["cache_path"]))
    client = OllamaClient(m["endpoint"], cache, num_ctx=m["num_ctx"], temperature=m["temperature"],
                          seed=cfg["seed"], think=m["think"])
    extractor = FactExtractor(client, m["extractor"], max_facts=mem["max_facts_per_session"],
                              registry_max_keys=mem["registry_max_keys"],
                              asst_chars=mem["assistant_trunc_chars_extract"])
    bus = SessionBus(infra["kafka_bootstrap"], infra["kafka_topic"])
    bus.ensure_topic()
    ctx = Ctx(cfg=cfg, client=client, extractor=extractor, store=FactStore(infra["pg_dsn"], m["embed_dim"]),
              wal=RedisWAL(infra["redis_url"]), bus=bus, seed=cfg["seed"])
    return ctx, Judge(client, m["judge"])


def select_instances(cfg: Config, n_smoke: int | None) -> list[PreparedInstance]:
    d = cfg["dataset"]
    ds = LongMemEvalDataset(repo_path(d["path"]), expected_sha256=d["sha256"])
    sample = stratified_sample(ds.load(), d["subset"], cfg["seed"])
    if n_smoke:
        sample = smoke_subset(sample, n_smoke)
    return [prepare(i, d["cutoff_mode"]) for i in sample]


def cells_for(cfg: Config, systems: list[str]) -> list[tuple[str, int, float]]:
    g = cfg["grid"]
    out = []
    for s in systems:
        if REGISTRY[s].write_mode == "sync":
            out.append((s, 0, 0.0))
        else:
            out.extend((s, lag, float(p)) for p in g["p_ooo"] for lag in g["lags"])
    return out


def run_instance(prep: PreparedInstance, ctx: Ctx, judge: Judge, run_id: str, cells, writer: JsonlWriter,
                 done: set[str], log) -> int:
    t0 = time.perf_counter()
    facts = ctx.extractor.extract_instance(prep)   # chronological, cached
    t_ext = time.perf_counter() - t0
    ev = prep.evidence_epochs()
    ev_keys = [{f.entity_key for f in facts.get(e, [])} for e in ev]
    diag = {
        "qid": prep.inst.qid, "qtype": prep.inst.qtype, "is_abstention": prep.inst.is_abstention,
        "n_sessions": prep.n, "n_facts": sum(len(v) for v in facts.values()),
        "evidence_epochs": ev, "evidence_fact_counts": [len(k) for k in ev_keys],
        "all_evidence_extracted": all(ev_keys) if ev_keys else None,
        # KU: does the latest evidence session write a key also written by an earlier evidence session?
        "update_key_collision": (bool(ev_keys[-1] & set().union(*ev_keys[:-1])) if len(ev_keys) >= 2 else None),
        "facts_by_evidence_epoch": {e: [f.to_dict() for f in facts.get(e, [])] for e in ev},
    }
    writer.append_side("extraction_diag.jsonl", diag)
    log(f"[{prep.inst.qid}] extraction ready ({prep.n} sessions, {t_ext:.1f}s, "
        f"evidence facts={diag['evidence_fact_counts']}, collision={diag['update_key_collision']})")
    written = 0
    ev_epochs = set(prep.evidence_epochs())
    latest_ev = max(ev_epochs) if ev_epochs else None
    for system, lag, p in cells:
        row_key = f"{prep.inst.qid}|{system}|{lag}|{p}"
        if row_key in done:
            continue
        ns = f"{run_id}:{row_key}"
        sysobj = REGISTRY[system](ctx)
        t1 = time.perf_counter()
        out = sysobj.run(prep, ns, lag, p)
        v = judge.judge(prep, out.answer)
        ing = out.ingest
        row = {
            "row_key": row_key, "run_id": run_id, "qid": prep.inst.qid, "qtype": prep.inst.qtype,
            "is_abstention": prep.inst.is_abstention, "system": system, "lag": lag, "p_ooo": p,
            "cutoff_mode": prep.cutoff_mode_applied, "n_sessions": prep.n,
            "question": prep.inst.question, "gold": prep.inst.answer, "answer": out.answer,
            "correct": v.correct, "abstained": v.abstained, "uses_superseded": v.uses_superseded,
            "judge_parse_errors": v.judge_parse_errors,
            "prompt_tokens": out.prompt_tokens, "server_prompt_tokens": out.server_prompt_tokens,
            "evidence_epochs": sorted(ev_epochs), "inflight_epochs": ing.inflight_epochs,
            "evidence_in_flight": latest_ev is not None and latest_ev in ing.inflight_epochs,
            "watermark": ing.watermark, "cas_rejections": ing.cas_rejections,
            "oocr_updated": ing.oocr_updated, "oocr_corrupted": ing.oocr_corrupted,
            "corrupted_keys": ing.corrupted_keys,
            "retrieved": [{k: f[k] for k in ("entity_key", "source_epoch", "fact_text")} for f in out.retrieved],
            "extra_context": out.extra_context, "reader_cached": out.reader_cached,
            "cell_wall_s": round(time.perf_counter() - t1, 3),
        }
        writer.append(row)
        done.add(row_key)
        written += 1
    log(f"[{prep.inst.qid}] done: {written} rows in {time.perf_counter() - t0:.1f}s")
    return written


def run(cfg: Config, run_id: str, systems: list[str], n_smoke: int | None = None) -> Path:
    run_dir = repo_path(cfg["runtime"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(run_dir / "results.jsonl")
    log_path = run_dir / "run.log"
    log_lock = threading.Lock()

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_lock, log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    ctx, judge = build_ctx(cfg)
    preps = select_instances(cfg, n_smoke)
    cells = cells_for(cfg, systems)
    done = writer.done_keys()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "seed": cfg["seed"], "models": cfg["models"], "grid": cfg["grid"],
        "cutoff_mode": cfg["dataset"]["cutoff_mode"], "systems": systems,
        "instances": [{"qid": p.inst.qid, "qtype": p.inst.qtype, "abs": p.inst.is_abstention,
                       "n_sessions": p.n, "cutoff": p.cutoff_mode_applied} for p in preps],
        "n_cells_per_instance": len(cells),
    }, indent=2), encoding="utf-8")
    log(f"run {run_id}: {len(preps)} instances x {len(cells)} cells; {len(done)} rows already done")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=cfg["runtime"]["workers"]) as ex:
        futs = {ex.submit(run_instance, p, ctx, judge, run_id, cells, writer, done, log): p for p in preps}
        for fut in as_completed(futs):
            try:
                fut.result()
            except Exception:  # keep other instances going; failure is logged and resumable
                log(f"[{futs[fut].inst.qid}] FAILED\n{traceback.format_exc()}")
    st = ctx.extractor.stats
    stats = {"wall_s": round(time.perf_counter() - t0, 1), "extraction": asdict(st),
             "cache_hits": ctx.client.cache.hits, "cache_misses": ctx.client.cache.misses,
             "live_chat_calls": ctx.client.stats.chat_calls, "live_chat_wall_s": round(ctx.client.stats.chat_wall_s, 1)}
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"finished: {json.dumps(stats)}")
    return run_dir
