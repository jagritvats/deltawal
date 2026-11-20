"""Stage C: Mem0 baselines on the 60 non-abstention knowledge-update questions of the held-out set.

Per question (post_update cutoff, T sessions, the last one is the update/evidence session):
  stage A = Mem0 after ingesting sessions 1..T-1   -> Mem0_async (lag=1), Mem0_deltawal (lag=1)
  stage B = stage A + session T                     -> Mem0_sync (lag=0)
Ingestion is done ONCE per question and Mem0 model and persisted on disk
(runs/mem0_qdrant/<llm>/<qid>_{A,B}/ with meta.json); reruns with another reader only search.
Every Mem0-internal LLM/embedding call goes through ModelRouter (guard -> SQLite cache -> HTTP).

Rows go to runs/<run_id>/results.jsonl (append-only, resumable by row_key) in the schema used by
judge_all.py / report_*.py; judging is done later (this script never judges).

  uv run python scripts/run_mem0.py --run-id m3mem0_luna6 --reader opencode-zen/gpt-6-luna --allow-cloud
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from epochmem.backends.mem0_wrapper import Mem0Backend, assemble_lines, reader_messages
from epochmem.cache import DiskCache
from epochmem.config import load_config, make_cloud_client, repo_path
from epochmem.guard import set_allow_cloud
from epochmem.llm import ModelRouter, OllamaClient, provider_of
from epochmem.memory.wal_embed import embed_turns_only
from epochmem.runner import JsonlWriter, select_instances
from epochmem.store.wal import RedisWAL
from epochmem.systems.base import READER_SYSTEM, MemorySystem
from epochmem.systems.deltawal import PRECEDENCE, gate_wal
from epochmem.tokens import count_messages

PAST = re.compile(r"\b(initial|initially|originally|first|before|previous|previously|used to|at first|earlier|prior|"
                  r"than i did|than before|switch|switched)\b", re.I)          # = scripts/report_controls.py
SYSTEMS = [("Mem0_async", 1), ("Mem0_deltawal", 1), ("Mem0_sync", 0)]


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model.split("/", 1)[-1])


def ku_instances(cfg, qids=None, limit=None):
    preps = [p for p in select_instances(cfg, None)
             if p.inst.qtype == "knowledge-update" and not p.inst.is_abstention]
    if qids:
        want = set(qids)
        preps = [p for p in preps if p.inst.qid in want]
        missing = want - {p.inst.qid for p in preps}
        if missing:
            raise SystemExit(f"not non-abstention KU held-out qids: {sorted(missing)}")
    return preps[:limit] if limit else preps


def make_router(cfg, models: list[str], session: str):
    m = cfg["models"]
    cache = DiskCache(repo_path(cfg["runtime"]["cache_path"]))
    local = OllamaClient(m["endpoint"], cache, num_ctx=m["num_ctx"], temperature=m["temperature"],
                         seed=cfg["seed"], think=m["think"])
    provs = {provider_of(x) for x in models} - {None}
    clouds = {p: make_cloud_client(cfg, cache, session_id=session, provider=p) for p in sorted(provs)}
    return ModelRouter(local, clouds, min_cloud_max_tokens=(cfg.get("cloud") or {}).get("min_max_tokens", 2048))


# ------------------------------------------------------------------ ingestion (once per question)
def ingest_stage(be: Mem0Backend, ns: str, sessions, log, base_ns: str | None = None) -> dict:
    meta_p = be.path(ns) / "meta.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text(encoding="utf-8"))
    if base_ns is None:
        be.reset(ns)
    else:
        be.snapshot(base_ns, ns)
    per = []
    for ss in sessions:
        r = be.add_session(ns, ss.session.turns, ss.epoch, ss.session.date, ss.session.session_id)
        per.append(r)
        i = r["internal"]
        log(f"    {ns} epoch={ss.epoch:>2} turns={r['n_turns']:>2} wall={r['wall_s']:6.1f}s calls={i['calls']} "
            f"live={i['live_calls']} tok={i['prompt_tokens']}+{i['completion_tokens']} pf={i['parse_failures']} "
            f"ev={r['events']}")
    hc = be.history_counts(ns)
    be.close(ns)
    meta = {"ns": ns, "done": True, "mem0_version": be.version, "llm": be.llm_model,
            "epochs": [ss.epoch for ss in sessions], "sessions": per,
            "history_totals": hc["totals"], "n_memories_ever": hc["n_memories_ever"], "n_live": hc["n_live"],
            "n_updated_memories": hc["n_updated_memories"], "per_memory": hc["per_memory"]}
    meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def stage_internal(*metas) -> dict:
    keys = ("calls", "live_calls", "cached_calls", "parse_failures", "prompt_tokens", "completion_tokens",
            "embed_calls", "mem0_errors")
    out = {k: 0 for k in keys}
    out["live_wall_s"] = 0.0
    for meta in metas:
        for s in meta["sessions"]:
            for k in keys:
                out[k] += s["internal"].get(k, 0)
            out["live_wall_s"] += s["internal"].get("live_wall_s", 0.0)
    out["live_wall_s"] = round(out["live_wall_s"], 2)
    out["n_sessions"] = sum(len(m["sessions"]) for m in metas)
    return out


_WORD = re.compile(r"[a-z0-9]+")


def _words(t: str) -> list[str]:
    return _WORD.findall((t or "").lower())


def gold_in_text(gold: str, text: str) -> bool | None:
    """Heuristic: is the (old-value) gold answer present in a memory text? None = not checkable
    (yes/no or long free-text golds). Short golds: word-sequence containment; numbers must all appear."""
    g = _words(gold)
    if not g or len(g) > 6 or g[0] in ("yes", "no", "you"):
        return None
    t = _words(text)
    nums = [w for w in g if w.isdigit()]
    if nums:
        return all(n in t for n in nums)
    return any(t[i:i + len(g)] == g for i in range(len(t) - len(g) + 1))


def ku_past_record(prep, meta_a, meta_b) -> dict:
    """Evidence for 'does Mem0's UPDATE erase history'. Every ADD/UPDATE/DELETE is tagged with the
    epoch that produced it. A memory 'holds an older evidence write' if it was ADDed or UPDATEd during an
    earlier evidence session; we flag its later UPDATE (in place: the store keeps only the new text, the
    old text survives only in Mem0's history table) or DELETE, report how many old words survive in the
    new text, and check (heuristically) whether the gold old value is still in the live store."""
    ev = prep.evidence_epochs()
    hist = [h for m in (meta_a, meta_b) for s in m["sessions"] for h in s["history"]]
    ev_written: dict[str, int] = {}
    live: dict[str, str | None] = {}
    for h in hist:
        if h["event"] in ("ADD", "UPDATE") and h["epoch"] in ev[:-1]:
            ev_written.setdefault(h["memory_id"], h["epoch"])
        live[h["memory_id"]] = h["new"] if h["event"] != "DELETE" else None
    by_ev = {e: {k: sum(1 for h in hist if h["epoch"] == e and h["event"] == k) for k in ("ADD", "UPDATE", "DELETE")}
             for e in ev}
    overwrites = []
    for h in hist:
        w = ev_written.get(h["memory_id"])
        if h["event"] in ("UPDATE", "DELETE") and w is not None and h["epoch"] > w:
            old_w, new_w = set(_words(h["old"])), set(_words(h["new"]))
            overwrites.append({**h, "evidence_write_epoch": w,
                               "old_words_retained": round(len(old_w & new_w) / max(1, len(old_w)), 3),
                               "gold_in_old": gold_in_text(prep.inst.answer, h["old"] or ""),
                               "gold_in_new": gold_in_text(prep.inst.answer, h["new"] or "")})
    final = [h for h in hist if h["epoch"] == prep.stream[-1].epoch]
    live_texts = [t for t in live.values() if t]
    g_live = [gold_in_text(prep.inst.answer, t) for t in live_texts]
    g_hist = [gold_in_text(prep.inst.answer, h["old"] or "") for h in hist if h["event"] in ("UPDATE", "DELETE")]
    checkable = gold_in_text(prep.inst.answer, "") is not None
    return {"qid": prep.inst.qid, "question": prep.inst.question, "gold": prep.inst.answer, "evidence_epochs": ev,
            "superseded_evidence": prep.superseded_evidence(300),
            "events_by_evidence_epoch": by_ev,
            "final_session_events": final,
            "older_evidence_memory_overwritten": overwrites,
            "old_value_updated_in_place": any(h["event"] == "UPDATE" for h in overwrites),
            "old_value_deleted": any(h["event"] == "DELETE" for h in overwrites),
            "gold_old_value_in_live_store": any(bool(x) for x in g_live) if checkable else None,
            "gold_old_value_only_in_history": (not any(bool(x) for x in g_live) and any(bool(x) for x in g_hist))
            if checkable else None,
            "totals_A": meta_a["history_totals"], "totals_B": meta_b["history_totals"]}


# ------------------------------------------------------------------ read paths
def run_question(prep, cfg, be, router, wal, writer, done, run_id, reader, log) -> int:
    qid = prep.inst.qid
    keys = {s: f"{qid}|{s}|{lag}|0.0" for s, lag in SYSTEMS}
    side = writer.path.parent / "mem0_ku_past.jsonl"
    past_done = side.exists() and any(json.loads(l)["qid"] == qid for l in side.open(encoding="utf-8") if l.strip())
    if all(k in done for k in keys.values()) and (past_done or not PAST.search(prep.inst.question)):
        return 0
    T = prep.n
    t0 = time.perf_counter()
    meta_a = ingest_stage(be, f"{qid}_A", prep.stream[:-1], log)
    meta_b = ingest_stage(be, f"{qid}_B", prep.stream[-1:], log, base_ns=f"{qid}_A")
    log(f"[{qid}] ingestion ready (T={T}, A={meta_a['history_totals']}, B={meta_b['history_totals']}, "
        f"{time.perf_counter() - t0:.1f}s)")
    if PAST.search(prep.inst.question) and not past_done:
        rec = ku_past_record(prep, meta_a, meta_b)
        with side.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    m, s, mem = cfg["models"], cfg["systems"], cfg["mem0"]
    k = int(mem.get("top_k", cfg["memory"]["top_k_facts"]))
    prec_base = PRECEDENCE if mem.get("precedence_in_baselines", True) else None
    qblock = MemorySystem.question_block(prep)
    ev = sorted(prep.evidence_epochs())
    qemb = router.embed(m["embedder"], [m["query_instruction"] + prep.inst.question])[0]
    written = 0
    for system, lag in SYSTEMS:
        row_key = keys[system]
        if row_key in done:
            continue
        ns = f"{qid}_A" if lag == 1 else f"{qid}_B"
        snap_meta = meta_a if lag == 1 else meta_b
        t1 = time.perf_counter()
        mems = be.search(ns, prep.inst.question, k)
        diag = {"mem0_ns": ns, "mem0_scores": [round(x["score"], 4) for x in mems]}
        gated = []
        if system == "Mem0_deltawal":
            wns = f"mem0:{run_id}:{qid}"
            wal.reset(wns)
            for ss in prep.stream:                         # write-through of every session's raw turns
                e = wal.next_epoch(wns)
                assert e == ss.epoch, (e, ss.epoch)
                wal.append_session(wns, e, ss)
            for ss in prep.stream[:-1]:                    # sessions 1..T-1 are committed to Mem0 (stage A)
                wal.mark_applied(wns, ss.epoch)
            committed, delta = wal.snapshot(wns)
            missing = [t for t in delta if t.get("embedding") is None]
            if missing:
                wal.store_embeddings(wns, embed_turns_only(lambda xs: router.embed(m["embedder"], xs), missing, s))
            gated = gate_wal(qemb, delta, s["wal_sim_threshold"], s["b4_delta_top_m"])
            wal.reset(wns)
            diag.update({"committed_epoch": committed, "n_uncommitted_turns": len(delta), "n_gated": len(gated),
                         "lazy_embedded_turns": len(missing),
                         "gated": [(t["epoch"], t["turn"], round(t["sim"], 4)) for t in gated],
                         "gated_turns": [{"epoch": t["epoch"], "turn": t["turn"], "role": t["role"],
                                          "content": t["content"][:600]} for t in gated]})
            lines = assemble_lines(mems, "OLDER MEMORY", gated, s["raw_turn_user_chars"], s["raw_turn_assistant_chars"])
            msgs = reader_messages(lines, qblock, READER_SYSTEM, PRECEDENCE)
        else:
            lines = assemble_lines(mems, "MEMORY")
            msgs = reader_messages(lines, qblock, READER_SYSTEM, prec_base)
        read_ms = 1000 * (time.perf_counter() - t1)
        res = router.chat(reader, msgs, max_tokens=256)
        inflight = [prep.stream[-1].epoch] if lag == 1 else []
        internal = stage_internal(meta_a) if lag == 1 else stage_internal(meta_a, meta_b)
        row = {
            "row_key": row_key, "run_id": run_id, "qid": qid, "qtype": prep.inst.qtype,
            "is_abstention": prep.inst.is_abstention, "system": system, "lag": lag, "p_ooo": 0.0,
            "cutoff_mode": prep.cutoff_mode_applied, "n_sessions": T,
            "question": prep.inst.question, "gold": prep.inst.answer, "answer": res.content.strip(),
            "correct": None, "abstained": None, "uses_superseded": None, "judge_parse_errors": None,
            "prompt_tokens": count_messages(msgs), "server_prompt_tokens": res.prompt_eval_count,
            "evidence_epochs": ev, "inflight_epochs": inflight, "evidence_in_flight": bool(ev) and ev[-1] in inflight,
            "watermark": T - 1 if lag == 1 else None, "cas_rejections": 0,
            "oocr_updated": 0, "oocr_corrupted": 0, "missed_merge": None, "reader_model": reader,
            "judge_model": None, "corrupted_keys": [],
            "retrieved": [{"entity_key": f"mem0:{x['id']}", "source_epoch": x["source_epoch"], "fact_text": x["memory"]}
                          for x in mems],
            "extra_context": [{"epoch": t["epoch"], "turn": t["turn"], "role": t["role"], "sim": t["sim"]} for t in gated],
            "reader_cached": res.cached, "read_path_ms": round(read_ms, 3), "read_llm_ms": 0.0,
            "read_diag": diag, "n_suppressed": 0, "b3_tail_turns": None,
            "cell_wall_s": round(time.perf_counter() - t1, 3),
            "memory_backend": f"mem0ai=={be.version}", "mem0_llm": be.llm_model,
            "mem0_history_counts": snap_meta["history_totals"] if lag == 1 else meta_b["history_totals"],
            "mem0_n_memories_live": snap_meta["n_live"], "mem0_n_updated_memories": snap_meta["n_updated_memories"],
            "mem0_internal_calls": internal["calls"], "mem0_parse_failures": internal["parse_failures"],
            "mem0_internal": internal,
        }
        writer.append(row)
        done.add(row_key)
        written += 1
    be.close(f"{qid}_A")
    be.close(f"{qid}_B")
    return written


def reingest_check(be: Mem0Backend, prep, log) -> dict:
    """Re-run stage A from scratch into a scratch namespace: must be all cache hits, same history."""
    ns = f"_reingest_{prep.inst.qid}_A"
    be.reset(ns)
    meta = ingest_stage(be, ns, prep.stream[:-1], log)
    ref = json.loads((be.path(f"{prep.inst.qid}_A") / "meta.json").read_text(encoding="utf-8"))
    out = {"qid": prep.inst.qid, "internal": stage_internal(meta), "history_totals": meta["history_totals"],
           "ref_history_totals": ref["history_totals"],
           "same_history_texts": [[(h["event"], h["old"], h["new"]) for h in s["history"]] for s in meta["sessions"]]
           == [[(h["event"], h["old"], h["new"]) for h in s["history"]] for s in ref["sessions"]]}
    be.reset(ns)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m3_mem0.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--reader", required=True, help="gemma4:12b-it-qat | opencode-zen/gpt-6-luna | opencode-zen/glm-5.3")
    ap.add_argument("--mem0-llm", default=None, help="override mem0.llm (fallback: opencode-zen/glm-5.3-flash)")
    ap.add_argument("--qids", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget-usd", type=float, default=None, help="override the opencode-zen per-process budget")
    ap.add_argument("--ingest-only", action="store_true", help="build Mem0 snapshots, no reader calls")
    ap.add_argument("--reingest-check", action="store_true", help="verify re-ingestion is fully cached")
    ap.add_argument("--systems", nargs="+", default=None, help="subset of Mem0_async Mem0_deltawal Mem0_sync")
    ap.add_argument("--ingest-workers", type=int, default=1,
                    help="with --ingest-only: questions ingested in parallel threads inside this one process")
    ap.add_argument("--no-judge", action="store_true", default=True, help="(always) rows are judged later by judge_all.py")
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    args = ap.parse_args()
    global SYSTEMS
    if args.systems:
        SYSTEMS = [(n, l) for n, l in SYSTEMS if n in set(args.systems)]
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    cfg["models"]["reader"] = args.reader
    mem = cfg["mem0"]
    llm = args.mem0_llm or mem["llm"]
    if args.budget_usd is not None:
        cfg["cloud"]["providers"]["opencode-zen"]["budget_usd"] = args.budget_usd
    run_dir = repo_path(cfg["runtime"]["runs_dir"]) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(run_dir / "results.jsonl")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with (run_dir / "run.log").open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    router = make_router(cfg, [args.reader, llm], f"ses_epochmem_{args.run_id}")
    m = cfg["models"]
    be = Mem0Backend(router, llm, m["embedder"], m["query_instruction"], m["embed_dim"],
                     repo_path(mem["store_dir"]) / f"{slug(llm)}_mem0-{mem['version']}",
                     user_chars=2000, asst_chars=cfg["memory"]["assistant_trunc_chars_extract"],
                     max_tokens=int(mem.get("internal_max_tokens", 2048)))
    preps = ku_instances(cfg, args.qids, args.limit)
    if args.reingest_check:
        for p in preps:
            log(json.dumps(reingest_check(be, p, log)))
        return 0
    wal = RedisWAL(cfg["infra"]["redis_url"], trim=cfg["infra"].get("wal_trim", True), dim=m["embed_dim"])
    done = writer.done_keys()
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": args.run_id, "seed": cfg["seed"], "reader": args.reader, "mem0_llm": llm,
        "mem0_version": be.version, "store": str(be.root), "systems": SYSTEMS, "config": args.config,
        "instances": [{"qid": p.inst.qid, "n_sessions": p.n, "evidence": p.evidence_epochs(),
                       "ku_past": bool(PAST.search(p.inst.question))} for p in preps]}, indent=2), encoding="utf-8")
    log(f"run {args.run_id}: {len(preps)} KU questions x {len(SYSTEMS)} systems; reader={args.reader} mem0_llm={llm}; "
        f"{len(done)} rows already done")
    t0 = time.perf_counter()
    failures = 0
    if args.ingest_only and args.ingest_workers > 1:
        # In-process parallel ingestion (ONE local-Ollama job): one Mem0Backend per worker thread so that
        # close_all() only closes that thread's stores; router + SQLite cache are shared and thread-safe.
        import threading
        from concurrent.futures import ThreadPoolExecutor
        tls, backends = threading.local(), []

        def worker_be():
            if not hasattr(tls, "be"):
                tls.be = Mem0Backend(router, llm, m["embedder"], m["query_instruction"], m["embed_dim"],
                                     repo_path(mem["store_dir"]) / f"{slug(llm)}_mem0-{mem['version']}",
                                     user_chars=2000, asst_chars=cfg["memory"]["assistant_trunc_chars_extract"],
                                     max_tokens=int(mem.get("internal_max_tokens", 2048)))
                backends.append(tls.be)
            return tls.be

        def ingest_one(p):
            b = worker_be()
            try:
                ingest_stage(b, f"{p.inst.qid}_A", p.stream[:-1], log)
                ingest_stage(b, f"{p.inst.qid}_B", p.stream[-1:], log, base_ns=f"{p.inst.qid}_A")
                log(f"[{p.inst.qid}] ingested")
                return 0
            except Exception as e:
                import traceback
                log(f"[{p.inst.qid}] FAILED {e!r}\n{traceback.format_exc()}")
                return 1
            finally:
                b.close_all()

        with ThreadPoolExecutor(args.ingest_workers) as ex:
            failures = sum(ex.map(ingest_one, preps))
        merged = {}
        for b in backends:
            for k, v in b.stats.snapshot().items():
                merged[k] = merged.get(k, 0) + v if isinstance(v, (int, float)) else v
        clouds = router.clouds
        stats = {"wall_s": round(time.perf_counter() - t0, 1), "failures": failures, "mem0_internal": merged,
                 "ingest_workers": args.ingest_workers,
                 "cloud_usage_by_model": {mm: u for c in clouds.values() for mm, u in c.usage_by_model.items()},
                 "cloud_spent_est_usd": round(sum(c.spent_usd() for c in clouds.values()), 4)}
        (run_dir / "stats_ingest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        log(f"finished ingest: {json.dumps({k: v for k, v in stats.items() if k != 'mem0_internal'})}")
        return 1 if failures else 0
    for p in preps:
        try:
            if args.ingest_only:
                ingest_stage(be, f"{p.inst.qid}_A", p.stream[:-1], log)
                ingest_stage(be, f"{p.inst.qid}_B", p.stream[-1:], log, base_ns=f"{p.inst.qid}_A")
                be.close_all()
                log(f"[{p.inst.qid}] ingested")
            else:
                n = run_question(p, cfg, be, router, wal, writer, done, args.run_id, args.reader, log)
                log(f"[{p.inst.qid}] {n} rows")
        except Exception as e:  # keep going; resumable
            failures += 1
            be.close_all()
            import traceback
            log(f"[{p.inst.qid}] FAILED {e!r}\n{traceback.format_exc()}")
            if e.__class__.__name__ == "BudgetExceeded":
                break
    clouds = router.clouds
    stats = {"wall_s": round(time.perf_counter() - t0, 1), "failures": failures,
             "mem0_internal": be.stats.snapshot(),
             "cache_hits": router.cache.hits, "cache_misses": router.cache.misses,
             "cloud_usage_by_model": {mm: u for c in clouds.values() for mm, u in c.usage_by_model.items()},
             "cloud_spent_est_usd": round(sum(c.spent_usd() for c in clouds.values()), 4)}
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(f"finished: {json.dumps({k: v for k, v in stats.items() if k != 'mem0_internal'})}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
