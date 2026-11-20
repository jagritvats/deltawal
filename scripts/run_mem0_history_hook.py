"""Read-only Mem0 ablation: Mem0_sync (lag 0) + a history hook, on the 13 KU-past held-out questions.

For each memory Mem0 retrieves from the persisted Stage C store (`<qid>_B`: sessions 1..T ingested), the hook
calls `Memory.history(memory_id)` and renders every earlier version of that memory (ADD text, then each
UPDATE's text) in chronological order alongside the current value. Epochs of the versions come from the
ingestion-time event log in meta.json (Mem0's history table only has wall-clock timestamps).

No ingestion and no writes to the Stage C store: each `<qid>_B` directory is copied to a scratch root
first and the copy is opened and deleted afterwards. Rows use the run_mem0.py schema and are judged by
judge_all.py.

  uv run python scripts/run_mem0_history_hook.py --run-id m4mem0hist_luna6 --reader opencode-zen/gpt-6-luna --allow-cloud
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time

from epochmem.backends.mem0_wrapper import Mem0Backend, reader_messages
from epochmem.config import load_config, repo_path
from epochmem.guard import set_allow_cloud
from epochmem.runner import JsonlWriter
from epochmem.systems.base import READER_SYSTEM, MemorySystem
from epochmem.systems.deltawal import PRECEDENCE
from epochmem.tokens import count_messages

sys.path.insert(0, str(repo_path("scripts")))
from run_mem0 import PAST, ku_instances, make_router, slug  # noqa: E402

SYSTEM = "Mem0_sync_history"


def versions(mem_id: str, hist: list[dict], events: list[dict]) -> list[tuple[int | None, str]]:
    """Chronological (epoch, text) versions of one memory from Memory.history() rows; epochs are taken from
    the ingestion event log (same memory_id, event and new text), in order."""
    ev = [e for e in events if e["memory_id"] == mem_id]
    out, j = [], 0
    for h in hist:
        if h["event"] not in ("ADD", "UPDATE"):
            continue
        epoch = None
        for k in range(j, len(ev)):
            if ev[k]["event"] == h["event"] and ev[k]["new"] == h["new_memory"]:
                epoch, j = ev[k]["epoch"], k + 1
                break
        out.append((epoch, h["new_memory"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m3_mem0.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--reader", required=True)
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    m, mem = cfg["models"], cfg["mem0"]
    src_root = repo_path(mem["store_dir"]) / f"{slug(mem['llm'])}_mem0-{mem['version']}"
    scratch = repo_path("runs") / args.run_id / "_store_copy"
    run_dir = repo_path("runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = JsonlWriter(run_dir / "results.jsonl")
    done = writer.done_keys()
    router = make_router(cfg, [args.reader], f"ses_epochmem_{args.run_id}")
    be = Mem0Backend(router, mem["llm"], m["embedder"], m["query_instruction"], m["embed_dim"], scratch,
                     user_chars=2000, asst_chars=cfg["memory"]["assistant_trunc_chars_extract"])
    preps = [p for p in ku_instances(cfg) if PAST.search(p.inst.question)]
    k = int(mem.get("top_k", cfg["memory"]["top_k_facts"]))
    prec = PRECEDENCE if mem.get("precedence_in_baselines", True) else None
    date_of = None
    n_written = 0
    for p in preps:
        qid = p.inst.qid
        row_key = f"{qid}|{SYSTEM}|0|0.0"
        if row_key in done:
            continue
        src = src_root / f"{qid}_B"
        if not (src / "history.db").exists():
            raise SystemExit(f"persisted Stage C store missing for {qid}: {src} (not re-ingesting)")
        shutil.rmtree(scratch / f"{qid}_B", ignore_errors=True)
        shutil.copytree(src, scratch / f"{qid}_B")
        meta_a = json.loads((src_root / f"{qid}_A" / "meta.json").read_text(encoding="utf-8"))
        meta_b = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        events = [h for mt in (meta_a, meta_b) for s in mt["sessions"] for h in s["history"]]
        date_of = {ss.epoch: ss.session.date for ss in p.stream}
        t1 = time.perf_counter()
        mems = be.search(f"{qid}_B", p.inst.question, k)
        mm = be._mem(f"{qid}_B")
        items, n_prior = [], 0
        for x in mems:
            vs = versions(x["id"], mm.history(x["id"]), events)
            for e, text in vs[:-1]:                      # earlier versions (the last one is the current value)
                items.append(((e if e is not None else -1), 0,
                              f"[MEMORY (earlier version) | epoch={e} | {date_of.get(e, '')}] {text}"))
                n_prior += 1
            se = x["source_epoch"] if x["source_epoch"] is not None else -1
            items.append((se, 1, f"[MEMORY | epoch={x['source_epoch']} | {x['session_date']}] {x['memory']}"))
        lines = [t[2] for t in sorted(items, key=lambda t: (t[0], t[1]))]
        msgs = reader_messages(lines, MemorySystem.question_block(p), READER_SYSTEM, prec)
        read_ms = 1000 * (time.perf_counter() - t1)
        res = router.chat(args.reader, msgs, max_tokens=256)
        be.close(f"{qid}_B")
        shutil.rmtree(scratch / f"{qid}_B", ignore_errors=True)
        ev = sorted(p.evidence_epochs())
        writer.append({
            "row_key": row_key, "run_id": args.run_id, "qid": qid, "qtype": p.inst.qtype,
            "is_abstention": p.inst.is_abstention, "system": SYSTEM, "lag": 0, "p_ooo": 0.0,
            "cutoff_mode": p.cutoff_mode_applied, "n_sessions": p.n,
            "question": p.inst.question, "gold": p.inst.answer, "answer": res.content.strip(),
            "correct": None, "abstained": None, "uses_superseded": None, "judge_parse_errors": None,
            "prompt_tokens": count_messages(msgs), "server_prompt_tokens": res.prompt_eval_count,
            "evidence_epochs": ev, "inflight_epochs": [], "evidence_in_flight": False,
            "watermark": None, "cas_rejections": 0, "oocr_updated": 0, "oocr_corrupted": 0, "missed_merge": None,
            "reader_model": args.reader, "judge_model": None, "corrupted_keys": [],
            "retrieved": [{"entity_key": f"mem0:{x['id']}", "source_epoch": x["source_epoch"], "fact_text": x["memory"]}
                          for x in mems],
            "extra_context": [], "reader_cached": res.cached, "read_path_ms": round(read_ms, 3), "read_llm_ms": 0.0,
            "read_diag": {"mem0_ns": f"{qid}_B", "n_prior_versions": n_prior, "lines": lines},
            "n_suppressed": 0, "b3_tail_turns": None, "cell_wall_s": round(time.perf_counter() - t1, 3),
            "memory_backend": f"mem0ai=={be.version}", "mem0_llm": mem["llm"],
        })
        n_written += 1
        print(f"[{qid}] prior_versions={n_prior} memories={len(mems)} cached={res.cached}", flush=True)
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"wrote {n_written} rows to {writer.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
