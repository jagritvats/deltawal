"""Cross-family re-judge of every prediction the paper reports.

Re-judges each unique (pass, question, normalized model answer) once with a judge from a second model family
(default: opencode-zen/claude-haiku-5-5), using the SAME judge prompts and parsing as the paper's judge
(`epochmem.eval.judge.Judge`: correctness prompt incl. the MemConflict static rubric, and behaviour prompt).
Instances are rebuilt with the same config the original `judge_all.py` pass used (README):

  p1  configs/m2_heldout.yaml       heldout + controls + unified   (m2ho_*, m2ctl_*, m2uni_*)
  p2  configs/m3_regression60.yaml  regression60                   (m3r60_*)
  p3  configs/m3_mem0.yaml          mem0 + mem0hist                (m3mem0_*, m4mem0hist_*)
  p4  configs/m3_memconflict.yaml   memconflict                    (m3mc_*)

Fidelity check (free): the reference judge (opencode-zen/qwen3.8-flash) is rebuilt exactly as judge_all.py
builds it with HTTP disabled; for every pair the raw answer variant whose reference prompts are cache hits is
found, and the cached reference verdict must reproduce the verdict stored in results_judged.jsonl. The
candidate judge then judges that exact raw answer, so both judges saw byte-identical prompts.

Budget: candidate spend is measured from the cache ledger (scripts/zen_spend.spend_since, candidate model only)
since the first run's start time (stored in <out>/state.json); the client's `budget_usd` is set to the cap minus
that, so the cap holds across resumed runs. HTTP 402 or BudgetExceeded stops new calls; verdicts are appended to
<out>/verdicts_pairs.jsonl as they arrive, so re-running resumes (and every finished call is a cache hit anyway).

Outputs (runs/ is gitignored; the row file carries no question/answer text):
  <out>/verdicts_pairs.jsonl   one line per unique pair: candidate verdict (+ the raw answer judged)
  <out>/rows.jsonl             one line per paper row, keyed like results_judged (run_id, row_key): qwen + haiku
  <out>/summary.json           counts, spend, agreement/kappa per group x field, headline comparisons

Usage:
  uv run python scripts/rejudge_crossfamily.py --allow-cloud --limit 50     # smoke + cost estimate
  uv run python scripts/rejudge_crossfamily.py --allow-cloud --workers 8    # everything (resumable)
  uv run python scripts/rejudge_crossfamily.py --analyze-only               # recompute summary from files
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_paper_assets import PAST, RUN_GROUPS  # noqa: E402  (read-only import: same run list + KU-past regex)
from judge_all import make_judge, normalize_answer  # noqa: E402
from zen_spend import spend_since  # noqa: E402

from epochmem.cache import DiskCache  # noqa: E402
from epochmem.config import load_config, make_cloud_client, repo_path  # noqa: E402
from epochmem.eval.judge import Judge  # noqa: E402
from epochmem.guard import set_allow_cloud  # noqa: E402
from epochmem.llm import BudgetExceeded, ModelRouter, OllamaClient  # noqa: E402
from epochmem.runner import select_instances  # noqa: E402

REF = "opencode-zen/qwen3.8-flash"
FIELDS = ("correct", "abstained", "uses_superseded")
PASSES = [("p1", "configs/m2_heldout.yaml", ["heldout", "controls", "unified"]),
          ("p2", "configs/m3_regression60.yaml", ["regression60"]),
          ("p3", "configs/m3_mem0.yaml", ["mem0", "mem0hist"]),
          ("p4", "configs/m3_memconflict.yaml", ["memconflict"])]
READERS3 = ["gemma4", "glm-5.3", "gpt-6-luna"]
MC_READERS = ["gemma4", "gpt-6-luna", "glm-5.3-flash"]


class NoNetwork(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------------ loading
def load_rows() -> list[dict]:
    out = []
    for pid, _cfg, groups in PASSES:
        for g in groups:
            for reader, run in RUN_GROUPS[g].items():
                for l in open(repo_path("runs") / run / "results_judged.jsonl", encoding="utf-8"):
                    if l.strip():
                        r = json.loads(l)
                        r["_pass"], r["_group"], r["_reader"], r["_run"] = pid, g, reader, run
                        out.append(r)
    return out


def pair_key(r: dict) -> tuple[str, str, str]:
    return r["_pass"], r["qid"], normalize_answer(r["answer"])


def vdict(v) -> dict:
    return {"correct": v.correct, "abstained": v.abstained, "uses_superseded": v.uses_superseded,
            "judge_parse_errors": v.judge_parse_errors}


def locate_reference(rows: list[dict], preps: dict) -> tuple[dict, dict]:
    """For each pair: the raw answer variant whose reference (qwen) prompts are cache hits, and the cached verdict.
    The reference client's HTTP is disabled, so this costs nothing and proves prompt identity."""
    variants: dict = defaultdict(list)
    for r in rows:
        k = pair_key(r)
        if r["answer"] not in variants[k]:
            variants[k].append(r["answer"])
    rep, refv = {}, {}
    judges = {}
    for pid, cfgp, _ in PASSES:
        cfg = load_config(cfgp)
        j, clouds = make_judge(cfg, REF, "ses_epochmem_judge_all")
        zen = clouds["opencode-zen"]

        def _blocked(*a, **kw):
            raise NoNetwork("reference network disabled (cache-only)")
        zen.http.post = _blocked
        zen.http.get = _blocked
        judges[pid] = (j, zen)
    miss = 0
    for k in sorted(variants):
        j, _ = judges[k[0]]
        for ans in variants[k]:
            try:
                refv[k] = vdict(j.judge(preps[(k[0], k[1])], ans))
                rep[k] = ans
                break
            except NoNetwork:
                continue
        else:
            miss += 1
            rep[k] = variants[k][0]
    for pid, (_, zen) in judges.items():
        assert zen.stats.chat_calls == 0 and not zen.usage_by_model, "reference client must not make calls"
    print(f"reference cache: {len(refv)}/{len(variants)} pairs located, {miss} not located", flush=True)
    return rep, refv


# ------------------------------------------------------------------------------------------------ judging
def candidate_judges(judge_model: str, budget: float, workers: int):
    prov, model = judge_model.split("/", 1)
    assert prov == "opencode-zen", "candidate judge must be on the pay-as-you-go Zen provider (never opencode-go)"
    out, clients = {}, []
    for pid, cfgp, _ in PASSES:
        cfg = copy.deepcopy(load_config(cfgp))
        pc = cfg["cloud"]["providers"]["opencode-zen"]
        pc["budget_usd"] = budget                       # per-process hard stop (cap minus ledger spend so far)
        pc["max_concurrency"] = workers
        pc["extra_body_by_model"] = {}                  # = judge_agreement.py's candidate body (cache-compatible)
        pc["prices"] = {model: [0.10, 0.50]}            # Zen list price (scripts/zen_spend.PRICES)
        m = cfg["models"]
        cache = DiskCache(repo_path(cfg["runtime"]["cache_path"]))
        local = OllamaClient(m["endpoint"], cache, num_ctx=m["num_ctx"], seed=cfg["seed"], think=m["think"])
        zen = make_cloud_client(cfg, cache, session_id="ses_epochmem_rejudge_crossfamily", provider="opencode-zen")
        router = ModelRouter(local, {"opencode-zen": zen}, (cfg.get("cloud") or {}).get("min_max_tokens", 2048))
        out[pid] = Judge(router, judge_model)
        clients.append(zen)
    return out, clients


def run_candidate(args, rows, preps, rep, out_dir: Path) -> dict:
    vpath = out_dir / "verdicts_pairs.jsonl"
    done = {}
    if vpath.exists():
        for l in open(vpath, encoding="utf-8"):
            o = json.loads(l)
            if o.get("judge_model") == args.judge:
                done[(o["pass"], o["qid"], o["key"])] = o
    pairs = sorted({pair_key(r) for r in rows})
    pending = [k for k in pairs if k not in done]
    random.Random(42).shuffle(pending)
    if args.limit:
        pending = pending[: args.limit]
    state_p = out_dir / "state.json"
    state = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {"since": time.time(),
                                                                                    "cap_usd": args.cap_usd}
    state_p.write_text(json.dumps(state, indent=1), encoding="utf-8")
    model = args.judge.split("/", 1)[1]
    ledger = spend_since(state["since"])["by_model"].get(model, {"usd": 0.0, "calls": 0})
    budget = round(args.cap_usd - ledger["usd"], 4)
    print(f"{len(pairs)} unique pairs, {len(done)} already judged, {len(pending)} to judge now; "
          f"ledger {model} since start ${ledger['usd']:.4f} ({ledger['calls']} calls) -> process budget ${budget:.4f}",
          flush=True)
    stats = {"pairs_total": len(pairs), "already_done": len(done), "attempted": 0, "judged": 0, "errors": 0,
             "stop_reason": None, "ledger_before_usd": ledger["usd"], "process_budget_usd": budget}
    if not pending:
        return stats
    if budget <= 0:
        stats["stop_reason"] = "cap reached before start"
        return stats
    judges, clients = candidate_judges(args.judge, budget, args.workers)
    stop = threading.Event()
    lock = threading.Lock()

    def work(k):
        if stop.is_set():
            return k, None, "skipped"
        try:
            return k, vdict(judges[k[0]].judge(preps[(k[0], k[1])], rep[k])), None
        except BudgetExceeded as e:
            stop.set()
            return k, None, f"BudgetExceeded: {e}"
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {str(e)[:300]}"
            if "HTTP 402" in msg:
                stop.set()
            return k, None, msg

    t0 = time.time()
    with open(vpath, "a", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, k) for k in pending]
        for i, f in enumerate(as_completed(futs), 1):
            k, v, err = f.result()
            if err == "skipped":
                continue
            stats["attempted"] += 1
            if v is None:
                stats["errors"] += 1
                if stats["errors"] <= 5 or "402" in err or "Budget" in err:
                    print(f"[error] {k[0]} {k[1]}: {err}", flush=True)
                if stop.is_set() and stats["stop_reason"] is None:
                    stats["stop_reason"] = err
                continue
            with lock:
                fh.write(json.dumps({"pass": k[0], "qid": k[1], "key": k[2], "answer": rep[k], **v,
                                     "judge_model": args.judge}, ensure_ascii=False) + "\n")
                fh.flush()
            stats["judged"] += 1
            if i % 250 == 0:
                spent = sum(c.spent_usd() for c in clients)
                print(f"  {i}/{len(pending)} done, process est. ${spent:.4f}, {time.time() - t0:.0f}s", flush=True)
    usage: dict = {}
    for c in clients:
        for m, u in c.usage_by_model.items():
            agg = usage.setdefault(m, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
            for kk in agg:
                agg[kk] += u[kk]
    stats["usage_this_process"] = usage
    stats["est_spend_this_process_usd"] = round(sum(c.spent_usd() for c in clients), 5)
    stats["ledger_after_usd"] = spend_since(state["since"])["by_model"].get(model, {"usd": 0.0})["usd"]
    stats["wall_s"] = round(time.time() - t0, 1)
    if stats["judged"]:
        per = stats["est_spend_this_process_usd"] / stats["judged"]
        remaining = len(pairs) - len(done) - stats["judged"]
        stats["est_usd_per_pair"] = round(per, 6)
        stats["est_usd_remaining_pairs"] = round(per * remaining, 4)
    return stats


# ------------------------------------------------------------------------------------------------ statistics
def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agree(a: list[bool], b: list[bool]) -> dict:
    n = len(a)
    k = kappa(a, b)
    return {"n": n, "agreement": round(sum(x == y for x, y in zip(a, b)) / n, 4) if n else None,
            "kappa": round(k, 4) if k is not None else None, "qwen_pos": sum(a), "haiku_pos": sum(b),
            "qwen_only": sum(x and not y for x, y in zip(a, b)), "haiku_only": sum(y and not x for x, y in zip(a, b))}


def stale(v: dict) -> bool:
    return (not v["correct"]) and bool(v["uses_superseded"])


def field(v: dict, f: str) -> bool:
    return stale(v) if f == "stale" else bool(v[f])


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(A: dict, B: dict) -> dict:
    qs = sorted(set(A) & set(B))
    n = len(qs)
    b = sum(1 for q in qs if A[q] and not B[q])
    c = sum(1 for q in qs if B[q] and not A[q])
    return {"n": n, "A": round(100 * sum(A[q] for q in qs) / n, 2) if n else None,
            "B": round(100 * sum(B[q] for q in qs) / n, 2) if n else None,
            "diff": round(100 * (b - c) / n, 2) if n else None, "A_only": b, "B_only": c,
            "p_mcnemar_exact": mcnemar_exact(b, c)}


def pooled(pairs: list[tuple[dict, dict]], iters: int = 4000, seed: int = 42) -> dict:
    """Pooled over readers: exact McNemar on summed discordant (reader, question) cells, plus a sign-flip
    permutation test with questions as clusters (scratchpad paired.py `pooled`)."""
    perq = defaultdict(list)
    b = c = tot = 0
    sa = sb = 0
    for A, B in pairs:
        for q in set(A) & set(B):
            perq[q].append(int(A[q]) - int(B[q]))
            b += A[q] and not B[q]
            c += B[q] and not A[q]
            sa += A[q]
            sb += B[q]
            tot += 1
    qs = sorted(perq)
    obs = sum(sum(perq[q]) for q in qs)
    rng = random.Random(seed)
    ge = sum(1 for _ in range(iters)
             if abs(sum(sum(perq[q]) * (1 if rng.random() < .5 else -1) for q in qs)) >= abs(obs) - 1e-9)
    return {"n_q": len(qs), "n_cells": tot, "A": round(100 * sa / tot, 2) if tot else None,
            "B": round(100 * sb / tot, 2) if tot else None, "diff": round(100 * obs / tot, 2) if tot else None,
            "A_only": int(b), "B_only": int(c), "p_mcnemar_exact": mcnemar_exact(int(b), int(c)),
            "p_perm_cluster": round((ge + 1) / (iters + 1), 4)}


# ------------------------------------------------------------------------------------------------ analysis
def is_ku(r):
    return r["qtype"] == "knowledge-update" and not r["is_abstention"]


def ku_past(r):
    return is_ku(r) and r["ku_past"]


def build_row_records(rows, refv, cand) -> list[dict]:
    out = []
    for r in rows:
        k = pair_key(r)
        q = {f: r[f] for f in FIELDS}
        h = cand.get(k)
        out.append({"run_id": r["_run"], "row_key": r["row_key"], "group": r["_group"], "reader": r["_reader"],
                    "qid": r["qid"], "qtype": r["qtype"], "is_abstention": r["is_abstention"], "system": r["system"],
                    "lag": r["lag"], "p_ooo": r["p_ooo"] or 0.0, "ku_past": bool(PAST.search(r["question"])),
                    "qwen": q, "qwen_ref_cache_matches_row": (refv.get(k) is not None and
                                                              all(refv[k][f] == q[f] for f in FIELDS)),
                    "haiku": {f: h[f] for f in FIELDS} if h else None})
    return out


def agreement_tables(recs, cand_pairs_ref) -> dict:
    out = {}
    groups = list(RUN_GROUPS)
    for g in groups + ["ALL"]:
        rs = [r for r in recs if r["haiku"] is not None and (g == "ALL" or r["group"] == g)]
        out[g] = {}
        for f in ("correct", "abstained", "uses_superseded", "stale"):
            out[g][f] = agree([field(r["qwen"], f) for r in rs], [field(r["haiku"], f) for r in rs])
        kus = [r for r in rs if is_ku(r)]
        if kus:
            out[g]["stale_KU_only"] = agree([stale(r["qwen"]) for r in kus], [stale(r["haiku"]) for r in kus])
    # per unique pair (each judged answer counted once)
    a = cand_pairs_ref
    out["ALL_unique_pairs"] = {f: agree([field(x, f) for x, _ in a], [field(y, f) for _, y in a])
                               for f in ("correct", "abstained", "uses_superseded", "stale")}
    return out


def cellmap(recs, judge, group, reader, system, lag, filt=lambda r: True, f="correct", qset=None) -> dict:
    out = {}
    for r in recs:
        if (r["group"] == group and r["reader"] == reader and r["system"] == system and r["lag"] == lag
                and r["p_ooo"] == 0.0 and filt(r) and (qset is None or r["qid"] in qset)):
            v = r[judge]
            if v is None:
                return {}
            out[r["qid"]] = field(v, f)
    return out


def comparisons(recs) -> list[dict]:
    a90 = {json.loads(l)["qid"]: json.loads(l)["verdict"] for l in open(repo_path("runs/memconflict_audit90.jsonl"),
                                                                          encoding="utf-8")}
    a90r2 = {json.loads(l)["qid"]: json.loads(l)["verdict"]
             for l in open(repo_path("runs/memconflict_audit90_r2.jsonl"), encoding="utf-8")}
    clean = {q for q, v in a90.items() if v == "clean" and a90r2.get(q) == "clean"}
    dyn = lambda r: r["qtype"] == "memconflict-dynamic"  # noqa: E731
    sta = lambda r: r["qtype"] == "memconflict-static"  # noqa: E731
    H, U = "heldout", "unified"
    # (name, readers, A spec, B spec, filter, field, qset, paper claim); spec = (group, system, lag)
    specs = [
        ("T1 KU acc: DWV lag1 vs B2 lag1", READERS3, (U, "deltawal_versioned", 1), (H, "B2_async", 1), is_ku,
         "correct", None, "DWV >> B2, every reader"),
        ("T1 KU stale: B2 lag1 vs DWV lag1", READERS3, (H, "B2_async", 1), (U, "deltawal_versioned", 1), is_ku,
         "stale", None, "B2 stale >> DWV, every reader"),
        ("T1 KU acc: DWV lag1 vs B1 lag0", READERS3, (U, "deltawal_versioned", 1), (H, "B1_sync", 0), is_ku,
         "correct", None, "DWV > B1 on KU for all three readers (gain traced to KU-past)"),
        ("T2a KU-past acc: DWV lag0 vs B1 lag0", READERS3, (U, "deltawal_versioned", 0), (H, "B1_sync", 0), ku_past,
         "correct", None, "DWV >> B1, non-overlapping CIs every reader"),
        ("Fig3a KU-past acc: DWV lag1 vs Shadow v2 lag1", READERS3, (U, "deltawal_versioned", 1),
         (H, "deltawal_shadow_v2", 1), ku_past, "correct", None, "read-side masking costs KU-past"),
        ("Fig3a KU-past acc: DWV lag1 vs Lazy lag1", READERS3, (U, "deltawal_versioned", 1), (H, "deltawal_lazy", 1),
         ku_past, "correct", None, "Lazy (drops superseded facts) below DWV"),
        ("T3 KU-past acc: Mem0 sync+history hook vs Mem0 sync", READERS3, ("mem0hist", "Mem0_sync_history", 0),
         ("mem0", "Mem0_sync", 0), ku_past, "correct", None, "hook raises KU-past to 100% every reader"),
        ("MC clean70 dynamic acc: DWV lag0 vs B1", MC_READERS, ("memconflict", "deltawal_versioned", 0),
         ("memconflict", "B1_sync", 0), dyn, "correct", clean, "DWV > B1 point estimate; intervals overlap"),
        ("MC clean70 static false-belief: DWV lag0 vs B1", MC_READERS, ("memconflict", "deltawal_versioned", 0),
         ("memconflict", "B1_sync", 0), sta, "stale", clean, "DWV higher for every reader (exploratory)"),
        ("R60 all acc: DWV lag0 vs B1", READERS3, ("regression60", "deltawal_versioned", 0),
         ("regression60", "B1_sync", 0), lambda r: True, "correct", None, "DWV >= B1 every reader; no regression"),
    ]
    out = []
    for name, readers, sa, sb, filt, f, qset, claim in specs:
        res = {"name": name, "claim": claim, "field": f}
        for judge in ("qwen", "haiku"):
            per, pp = {}, []
            for rd in readers:
                A = cellmap(recs, judge, sa[0], rd, sa[1], sa[2], filt, f, qset)
                B = cellmap(recs, judge, sb[0], rd, sb[1], sb[2], filt, f, qset)
                if A and B:
                    per[rd] = paired(A, B)
                    pp.append((A, B))
            res[judge] = {"per_reader": per, "pooled": pooled(pp) if pp else None}
        q, h = res["qwen"], res["haiku"]
        if h["pooled"] is None or len(h["per_reader"]) != len(q["per_reader"]):
            res["unchanged"] = None
        else:
            sgn = lambda x: (x > 0) - (x < 0)  # noqa: E731
            same_sign = all(sgn(q["per_reader"][rd]["diff"]) == sgn(h["per_reader"][rd]["diff"])
                            for rd in q["per_reader"]) and sgn(q["pooled"]["diff"]) == sgn(h["pooled"]["diff"])
            sig = lambda d: d["p_mcnemar_exact"] < 0.05  # noqa: E731
            same_sig_reader = all(sig(q["per_reader"][rd]) == sig(h["per_reader"][rd]) for rd in q["per_reader"])
            same_sig_pooled = (sig(q["pooled"]) == sig(h["pooled"]) and
                               (q["pooled"]["p_perm_cluster"] < .05) == (h["pooled"]["p_perm_cluster"] < .05))
            res["same_sign_all"] = same_sign
            res["same_significance_per_reader"] = same_sig_reader
            res["same_significance_pooled"] = same_sig_pooled
            res["unchanged"] = bool(same_sign and same_sig_pooled)
        out.append(res)
    return out


def analyze(rows, refv, out_dir: Path, run_stats: dict | None, judge_model: str) -> dict:
    cand = {}
    vpath = out_dir / "verdicts_pairs.jsonl"
    if vpath.exists():
        for l in open(vpath, encoding="utf-8"):
            o = json.loads(l)
            if o.get("judge_model") == judge_model:
                cand[(o["pass"], o["qid"], o["key"])] = o
    recs = build_row_records(rows, refv, cand)
    with open(out_dir / "rows.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    pairs = sorted({pair_key(r) for r in rows})
    ref_pairs = [(refv[k], cand[k]) for k in pairs if k in refv and k in cand]
    summary = {
        "judge_reference": REF, "judge_candidate": judge_model,
        "rows": len(recs), "rows_with_candidate": sum(r["haiku"] is not None for r in recs),
        "unique_pairs": len(pairs), "unique_pairs_judged": sum(k in cand for k in pairs),
        "reference_located_in_cache": len(refv),
        "rows_reference_cache_matches_stored_verdict": sum(r["qwen_ref_cache_matches_row"] for r in recs),
        "candidate_parse_errors": sum(cand[k]["judge_parse_errors"] for k in pairs if k in cand),
        "rows_by_group": {g: sum(r["group"] == g for r in recs) for g in RUN_GROUPS},
        "run": run_stats,
        "agreement": agreement_tables(recs, ref_pairs),
        "comparisons": comparisons(recs),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    return summary


def print_report(s: dict) -> None:
    print(f"\nrows {s['rows']} (with candidate {s['rows_with_candidate']}); unique pairs {s['unique_pairs']} "
          f"(judged {s['unique_pairs_judged']}); reference located {s['reference_located_in_cache']}; rows whose "
          f"cached reference verdict == stored verdict: {s['rows_reference_cache_matches_stored_verdict']}; "
          f"candidate parse errors {s['candidate_parse_errors']}")
    print("\nAgreement qwen vs candidate (agreement / kappa / n)")
    for g, d in s["agreement"].items():
        print(f"  {g:18s} " + "  ".join(f"{f}={v['agreement']}/{v['kappa']}/{v['n']}" for f, v in d.items()))
    print("\nHeadline comparisons (A-B, pp; exact McNemar p)")
    for c in s["comparisons"]:
        print(f"  {c['name']}  [unchanged={c.get('unchanged')}]")
        for j in ("qwen", "haiku"):
            per = c[j]["per_reader"]
            p = c[j]["pooled"]
            if p is None:
                print(f"    {j:5s} n/a")
                continue
            print(f"    {j:5s} " + " | ".join(f"{rd}: {d['A']:.1f} vs {d['B']:.1f} ({d['diff']:+.1f}, p={d['p_mcnemar_exact']:.3g})"
                                       for rd, d in per.items())
                  + f" || pooled {p['A']:.1f} vs {p['B']:.1f} ({p['diff']:+.1f}, McN p={p['p_mcnemar_exact']:.3g}, "
                    f"perm p={p['p_perm_cluster']:.3g})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="opencode-zen/claude-haiku-5-5")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="judge at most N pending pairs (smoke / cost estimate)")
    ap.add_argument("--cap-usd", type=float, default=1.20, help="hard cap on new candidate spend across all runs")
    ap.add_argument("--out", default="runs/rejudge_haiku")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    args = ap.parse_args()
    assert not args.judge.startswith("opencode-go/"), "opencode-go is never used here"
    assert args.workers <= 8, "moderate concurrency only (<= 8)"
    set_allow_cloud(args.allow_cloud)
    out_dir = repo_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    preps = {}
    for pid, cfgp, _ in PASSES:
        for p in select_instances(load_config(cfgp), None):
            preps[(pid, p.inst.qid)] = p
    missing = {(r["_pass"], r["qid"]) for r in rows} - set(preps)
    assert not missing, f"rows without a prepared instance: {sorted(missing)[:5]}"
    print(f"{len(rows)} rows, {len({pair_key(r) for r in rows})} unique (pass, qid, normalized answer) pairs", flush=True)
    rep, refv = locate_reference(rows, preps) if args.allow_cloud else ({}, {})
    if not args.allow_cloud:
        cached = out_dir / "reference_verdicts.json"
        if cached.exists():
            refv = {tuple(k.split("\t")): v for k, v in json.loads(cached.read_text(encoding="utf-8")).items()}
    else:
        (out_dir / "reference_verdicts.json").write_text(
            json.dumps({"\t".join(k): v for k, v in refv.items()}), encoding="utf-8")
    run_stats = None
    if not args.analyze_only:
        if not args.allow_cloud:
            raise SystemExit("judging needs --allow-cloud (or use --analyze-only)")
        run_stats = run_candidate(args, rows, preps, rep, out_dir)
        print(json.dumps(run_stats, indent=1), flush=True)
    s = analyze(rows, refv, out_dir, run_stats, args.judge)
    print_report(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
