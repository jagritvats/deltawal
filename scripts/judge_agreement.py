"""Judge-replacement check: agreement of cheaper candidate judges with the reference judge (qwen4-max).

Reference verdicts are NEVER re-requested from the network:
  * dev (20 q, 82 unique (qid, answer) pairs): read from runs/m2_dev20/rejudge_opencode-go_qwen4-max.jsonl
  * held-out (138 q): rebuilt from SQLite cache hits only -- the reference Judge is built exactly as
    scripts/judge_all.py builds it and its Go client's HTTP `post` is patched to raise, so a cache miss
    raises and the pair is skipped.

Candidates are judged through the normal guard -> cache -> client path on the opencode-zen provider
(config `budget_usd` hard stop, `prices` map). Metrics (exact agreement + Cohen's kappa):
  correct, abstained, stale_read = (not correct) and uses_superseded on knowledge-update non-abstention
  questions, confident_wrong = (not correct) and (not abstained).
Unique pairs are weighted once each; dev metrics are additionally reported weighted per row (820 rows).

Usage:
  uv run python scripts/judge_agreement.py --allow-cloud
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from judge_all import make_judge, normalize_answer  # noqa: E402

from epochmem.cache import DiskCache  # noqa: E402
from epochmem.config import load_config, make_cloud_client, repo_path  # noqa: E402
from epochmem.eval.judge import Judge  # noqa: E402
from epochmem.guard import set_allow_cloud  # noqa: E402
from epochmem.llm import BudgetExceeded, ModelRouter, OllamaClient  # noqa: E402
from epochmem.runner import select_instances  # noqa: E402

REF = "opencode-go/qwen4-max"
FIELDS = ("correct", "abstained", "uses_superseded")


class NoNetwork(RuntimeError):
    pass


def kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def metric(a: list[bool], b: list[bool]) -> dict:
    n = len(a)
    k = kappa(a, b)
    return {"n": n, "agreement": round(sum(x == y for x, y in zip(a, b)) / n, 4) if n else None,
            "kappa": round(k, 4) if k is not None else None,
            "ref_pos": sum(a), "cand_pos": sum(b),
            "ref_true_cand_false": sum(x and not y for x, y in zip(a, b)),
            "ref_false_cand_true": sum(y and not x for x, y in zip(a, b))}


def kappa_ci(a: list[bool], b: list[bool], n_boot: int = 2000, seed: int = 42) -> list | None:
    """Percentile bootstrap 95% CI for kappa (resampling units = unique pairs)."""
    n = len(a)
    if n == 0:
        return None
    rng = random.Random(seed)
    ks = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        k = kappa([a[i] for i in idx], [b[i] for i in idx])
        ks.append(k)
    ks.sort()
    return [round(ks[int(0.025 * n_boot)], 4), round(ks[int(0.975 * n_boot) - 1], 4)]


def derived(v: dict, ku_nonabs: bool) -> dict:
    return {"correct": v["correct"], "abstained": v["abstained"],
            "stale_read": (not v["correct"]) and v["uses_superseded"] if ku_nonabs else None,
            "uses_superseded_ku": v["uses_superseded"] if ku_nonabs else None,
            "confident_wrong": (not v["correct"]) and (not v["abstained"])}


def agreement(items: list[tuple[dict, dict, bool, int]]) -> dict:
    """items: (ref verdict, cand verdict, ku_nonabs, weight). Weight = number of rows a pair stands for."""
    out = {}
    for f in ("correct", "abstained", "stale_read", "uses_superseded_ku", "confident_wrong"):
        a, b = [], []
        for r, c, ku, w in items:
            dr, dc = derived(r, ku), derived(c, ku)
            if dr[f] is None:
                continue
            a += [dr[f]] * w
            b += [dc[f]] * w
        out[f] = metric(a, b)
        if all(w == 1 for *_, w in items):
            out[f]["kappa_ci95"] = kappa_ci(a, b)
    return out


def vdict(v) -> dict:
    return {"correct": v.correct, "abstained": v.abstained, "uses_superseded": v.uses_superseded,
            "judge_parse_errors": v.judge_parse_errors}


def judge_many(judge: Judge, preps: dict, pairs: list[tuple[str, str]], workers: int) -> dict:
    """Returns {(qid, answer): verdict dict | {"error": str}} (budget stop / call failure kept per pair)."""
    def work(pair):
        qid, ans = pair
        try:
            return pair, vdict(judge.judge(preps[qid], ans))
        except BudgetExceeded as e:
            return pair, {"error": f"BudgetExceeded: {e}"}
        except Exception as e:  # noqa: BLE001 -- recorded and reported, not fatal for the study
            return pair, {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(work, pairs))


def zen_judges(cfg, candidates: list[str], session: str):
    m = cfg["models"]
    cache = DiskCache(repo_path(cfg["runtime"]["cache_path"]))
    local = OllamaClient(m["endpoint"], cache, num_ctx=m["num_ctx"], seed=cfg["seed"], think=m["think"])
    zen = make_cloud_client(cfg, cache, session_id=session, provider="opencode-zen")
    router = ModelRouter(local, {"opencode-zen": zen}, (cfg.get("cloud") or {}).get("min_max_tokens", 2048))
    return {c: Judge(router, f"opencode-zen/{c}") for c in candidates}, zen


# ---------------------------------------------------------------- datasets
def load_dev(cfg, limit: int | None):
    preps = {p.inst.qid: p for p in select_instances(cfg, None)}
    rows = [json.loads(l) for l in open(repo_path("runs/m2_dev20/results.jsonl"), encoding="utf-8")]
    ref_rows = {}
    for l in open(repo_path("runs/m2_dev20/rejudge_opencode-go_qwen4-max.jsonl"), encoding="utf-8"):
        o = json.loads(l)
        ref_rows[o["row_key"]] = {**o["cloud"], "judge_parse_errors": o.get("cloud_parse_errors", 0)}
    ref, weight, conflicts = {}, {}, 0
    for r in rows:
        p = (r["qid"], r["answer"])
        v = ref_rows[r["row_key"]]
        if p in ref and any(ref[p][f] != v[f] for f in FIELDS):
            conflicts += 1
        ref.setdefault(p, v)
        weight[p] = weight.get(p, 0) + 1
    pairs = sorted(ref)
    if limit:
        pairs = pairs[:limit]
    return preps, pairs, ref, weight, {"rows": len(rows), "unique_pairs": len(ref), "ref_conflicts": conflicts}


def load_heldout(cfg, n_target: int, seed: int):
    preps = {p.inst.qid: p for p in select_instances(cfg, None)}
    pairs: dict = {}
    weight: dict = {}
    for run in ("m2ho_luna6", "m2ho_glm"):
        for l in open(repo_path("runs") / run / "results.jsonl", encoding="utf-8"):
            row = json.loads(l)
            k = (row["qid"], normalize_answer(row["answer"]))
            pairs.setdefault(k, row["answer"])
            weight[k] = weight.get(k, 0) + 1
    # reference judge exactly as judge_all.py builds it, network disabled
    ref_judge, clouds = make_judge(cfg, REF, "ses_epochmem_judge_all")
    go = clouds["opencode-go"]

    def _blocked(*a, **kw):
        raise NoNetwork("opencode-go network disabled (cache-only read)")
    go.http.post = _blocked
    go.http.get = _blocked
    hits, misses = {}, 0
    for k in sorted(pairs):
        try:
            hits[k] = vdict(ref_judge.judge(preps[k[0]], pairs[k]))
        except NoNetwork:
            misses += 1
    assert go.stats.chat_calls == 0 and not go.usage_by_model, "reference client must not have made calls"
    # stratified deterministic sample over cache hits
    rng = random.Random(seed)
    by: dict = {}
    for k in sorted(hits):
        by.setdefault(preps[k[0]].inst.stratum, []).append(k)
    picked: list = []
    if len(hits) <= n_target:
        picked = sorted(hits)
    else:
        strata = sorted(by)
        per = n_target // len(strata)
        for s in strata:
            picked += rng.sample(by[s], min(per, len(by[s])))
        rest = sorted(k for k in hits if k not in set(picked))
        rng.shuffle(rest)
        picked += rest[: n_target - len(picked)]
        picked = sorted(picked)
    info = {"unique_pairs_total": len(pairs), "ref_cache_hits": len(hits), "ref_cache_misses": misses,
            "hits_by_stratum": {s: len(v) for s, v in sorted(by.items())},
            "sampled": len(picked),
            "sampled_by_stratum": {s: sum(preps[k[0]].inst.stratum == s for k in picked) for s in sorted(by)}}
    # judge input uses the representative raw answer
    return preps, picked, pairs, hits, weight, info


# ---------------------------------------------------------------- main
def examples(items, preps, rep_answer, limit=6):
    out = []
    for p, r, c in items:
        diff = [f for f in FIELDS if r[f] != c[f]]
        if not diff:
            continue
        inst = preps[p[0]].inst
        out.append({"qid": p[0], "qtype": inst.qtype, "abs_q": inst.is_abstention, "fields": diff,
                    "question": inst.question[:160], "gold": inst.answer[:160], "answer": rep_answer(p)[:240],
                    "ref": {f: r[f] for f in FIELDS}, "cand": {f: c[f] for f in FIELDS}})
    return out[:limit], len(out)


def evaluate(name, preps, pairs, ref, cand, weight_fn, rep_answer):
    ok = [p for p in pairs if "error" not in cand[p]]
    ok_set = set(ok)
    errs = [p for p in pairs if "error" in cand[p]]
    ku = lambda p: preps[p[0]].inst.qtype == "knowledge-update" and not preps[p[0]].inst.is_abstention  # noqa: E731
    res = {"pairs_requested": len(pairs), "pairs_judged": len(ok), "call_errors": len(errs),
           "call_error_samples": [cand[p]["error"] for p in errs[:3]],
           "cand_parse_errors": sum(cand[p]["judge_parse_errors"] for p in ok),
           "cand_pairs_with_parse_errors": sum(cand[p]["judge_parse_errors"] > 0 for p in ok),
           "ref_parse_errors": sum(ref[p].get("judge_parse_errors", 0) for p in ok),
           "per_pair": agreement([(ref[p], cand[p], ku(p), 1) for p in ok]),
           "verdicts": [{"qid": p[0], "key": p[1][:120], "ref": [ref[p][f] for f in FIELDS],
                         "cand": [cand[p][f] for f in FIELDS] if p in ok_set else None} for p in pairs]}
    if weight_fn is not None:
        res["per_row"] = agreement([(ref[p], cand[p], ku(p), weight_fn(p)) for p in ok])
    ex, ndis = examples([(p, ref[p], cand[p]) for p in ok], preps, rep_answer)
    res["pairs_with_any_disagreement"] = ndis
    res["disagreement_examples"] = ex
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-config", default="configs/judge_check_dev.yaml")
    ap.add_argument("--heldout-config", default="configs/judge_check_heldout.yaml")
    ap.add_argument("--candidates", nargs="+", default=["qwen3.8-flash", "claude-haiku-5-5"])
    ap.add_argument("--datasets", nargs="+", default=["dev", "heldout"])
    ap.add_argument("--heldout-n", type=int, default=240)
    ap.add_argument("--dev-limit", type=int, default=None, help="probe: judge only the first N dev pairs")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/judge_agreement_check.json")
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)

    report = {"reference": REF, "candidates": args.candidates, "datasets": {}, "spend": {}}
    if "dev" in args.datasets:
        cfg = load_config(args.dev_config)
        preps, pairs, ref, weight, info = load_dev(cfg, args.dev_limit)
        judges, zen = zen_judges(cfg, args.candidates, "ses_epochmem_judge_agreement_dev")
        d = {"info": info, "config": args.dev_config,
             "extra_body_by_model": zen.extra_body_by_model, "by_candidate": {}}
        for c, j in judges.items():
            cand = judge_many(j, preps, pairs, args.workers)
            d["by_candidate"][c] = evaluate("dev", preps, pairs, ref, cand, weight.get, lambda p: p[1])
            print(f"[dev] {c}: spent so far ${zen.spent_usd():.4f}", flush=True)
        report["datasets"]["dev"] = d
        report["spend"]["dev"] = {"spent_usd": round(zen.spent_usd(), 5), "usage_by_model": zen.usage_by_model}
    if "heldout" in args.datasets:
        cfg = load_config(args.heldout_config)
        preps, picked, rep, ref, weight, info = load_heldout(cfg, args.heldout_n, cfg["seed"])
        print(f"[heldout] {json.dumps(info)}", flush=True)
        judges, zen = zen_judges(cfg, args.candidates, "ses_epochmem_judge_agreement_heldout")
        d = {"info": info, "config": args.heldout_config,
             "extra_body_by_model": zen.extra_body_by_model, "by_candidate": {}}
        for c, j in judges.items():
            # candidates judge the same representative raw answer judge_all.py sent to the reference
            raw = judge_many(j, preps, [(q, rep[(q, n)]) for q, n in picked], args.workers)
            cand = {(q, n): raw[(q, rep[(q, n)])] for q, n in picked}
            d["by_candidate"][c] = evaluate("heldout", preps, picked, ref, cand, weight.get, lambda p: rep[p])
            print(f"[heldout] {c}: spent so far ${zen.spent_usd():.4f}", flush=True)
        report["datasets"]["heldout"] = d
        report["spend"]["heldout"] = {"spent_usd": round(zen.spent_usd(), 5), "usage_by_model": zen.usage_by_model}
    rec = {}
    for c in args.candidates:
        ks = {f"{ds}.{f}": d["by_candidate"][c]["per_pair"][f]["kappa"]
              for ds, d in report["datasets"].items() for f in ("correct", "stale_read")}
        rec[c] = {"kappas_per_pair": ks,
                  "meets_rule_kappa_ge_0.95_correct_and_stale_read": all(k is not None and k >= 0.95
                                                                         for k in ks.values())}
    report["decision_rule"] = rec
    report["spend"]["total_usd_this_process"] = round(sum(s["spent_usd"] for s in report["spend"].values()), 5)
    report["note"] = ("spent_usd counts only uncached calls made by this process; re-runs hit the cache. "
                      "Reference verdicts: dev from rejudge file, held-out from SQLite cache hits only.")
    out = repo_path(args.out)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for ds, d in report["datasets"].items():
        for c, r in d["by_candidate"].items():
            pp = r["per_pair"]
            print(f"{ds:8s} {c:18s} n={r['pairs_judged']:4d} err={r['call_errors']} parse={r['cand_parse_errors']} "
                  + " ".join(f"{f}={pp[f]['agreement']}/k{pp[f]['kappa']}" for f in
                             ("correct", "abstained", "stale_read", "confident_wrong")))
    print(json.dumps(report["spend"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
