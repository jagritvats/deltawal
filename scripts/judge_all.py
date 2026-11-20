"""Deduplicated cross-run judge pass.

Collects every (question_id, normalized_answer) pair across the given runs, judges each unique
pair ONCE with a single cross-family judge, writes `results_judged.jsonl` next to each run's
`results.jsonl`, and exports a stratified 50-row human-audit CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from epochmem.cache import DiskCache
from epochmem.config import load_config, make_cloud_client, repo_path
from epochmem.eval.judge import Judge
from epochmem.guard import set_allow_cloud
from epochmem.llm import ModelRouter, OllamaClient, provider_of
from epochmem.runner import enforce_provider_caps, select_instances


def normalize_answer(a: str) -> str:
    a = (a or "").replace("**", "").replace("’", "'").replace("—", "-")
    a = re.sub(r"\s+", " ", a).strip().lower()
    return a.rstrip(" .!")


def make_judge(cfg, judge_model: str, session: str):
    m = cfg["models"]
    cache = DiskCache(repo_path(cfg["runtime"]["cache_path"]))
    local = OllamaClient(m["endpoint"], cache, num_ctx=m["num_ctx"], seed=cfg["seed"], think=m["think"])
    prov = provider_of(judge_model)
    clouds = {prov: make_cloud_client(cfg, cache, session_id=session, provider=prov)} if prov else {}
    router = ModelRouter(local, clouds, (cfg.get("cloud") or {}).get("min_max_tokens", 2048))
    return Judge(router, judge_model), clouds


def judge_pairs(pairs: dict, preps: dict, judge: Judge, workers: int) -> dict:
    """pairs: {(qid, norm): representative_answer} -> {(qid, norm): verdict dict}."""
    def work(item):
        (qid, norm), ans = item
        v = judge.judge(preps[qid], ans)
        return (qid, norm), {"correct": v.correct, "abstained": v.abstained,
                             "uses_superseded": v.uses_superseded, "judge_parse_errors": v.judge_parse_errors}
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(work, sorted(pairs.items())))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_heldout.yaml")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--judge", default="opencode-go/qwen4-max")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--audit-n", type=int, default=50)
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    ap.add_argument("--allow-go-scale", action="store_true", default=False)
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    cfg.setdefault("runtime", {})["allow_go_scale"] = args.allow_go_scale
    cfg["models"]["judge"] = args.judge
    preps = {p.inst.qid: p for p in select_instances(cfg, None)}
    enforce_provider_caps(cfg, len(preps))
    judge, clouds = make_judge(cfg, args.judge, "ses_epochmem_judge_all")

    runs = {r: [json.loads(l) for l in open(repo_path("runs") / r / "results.jsonl", encoding="utf-8")] for r in args.runs}
    pairs: dict = {}
    n_rows = 0
    for rows in runs.values():
        for row in rows:
            n_rows += 1
            pairs.setdefault((row["qid"], normalize_answer(row["answer"])), row["answer"])
    print(f"{n_rows} rows across {len(runs)} runs -> {len(pairs)} unique (qid, normalized answer) pairs", flush=True)
    verdicts = judge_pairs(pairs, preps, judge, args.workers)

    for run, rows in runs.items():
        with open(repo_path("runs") / run / "results_judged.jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                v = verdicts[(row["qid"], normalize_answer(row["answer"]))]
                fh.write(json.dumps({**row, **v, "judge_model": args.judge}, ensure_ascii=False) + "\n")

    # stratified human audit: stratum x verdict, over unique pairs
    rng = random.Random(cfg["seed"])
    cells: dict = {}
    for (qid, norm), ans in pairs.items():
        p = preps[qid]
        cells.setdefault((p.inst.stratum, verdicts[(qid, norm)]["correct"]), []).append((qid, ans))
    keys = sorted(cells, key=str)
    per = max(1, args.audit_n // len(keys))
    picked = []
    for k in keys:
        pool = sorted(cells[k])
        picked += [(k, x) for x in rng.sample(pool, min(per, len(pool)))]
    leftover = [(k, x) for k in keys for x in sorted(cells[k]) if (k, x) not in picked]
    rng.shuffle(leftover)
    picked += leftover[: max(0, args.audit_n - len(picked))]
    out = repo_path("runs/human_audit_50.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["audit_id", "qid", "stratum", "question", "gold_answer", "model_answer", "judge_correct",
                    "judge_abstained", "judge_uses_superseded", "human_correct", "human_abstained",
                    "human_uses_superseded", "notes"])
        for i, ((stratum, _), (qid, ans)) in enumerate(picked, 1):
            v = verdicts[(qid, normalize_answer(ans))]
            p = preps[qid]
            w.writerow([i, qid, stratum, p.inst.question, p.inst.answer, ans, v["correct"], v["abstained"],
                        v["uses_superseded"], "", "", "", ""])
    usage = {m: u for c in clouds.values() for m, u in c.usage_by_model.items()}
    summary = {"rows": n_rows, "unique_pairs": len(pairs), "judge": args.judge,
               "parse_errors": sum(v["judge_parse_errors"] for v in verdicts.values()),
               "usage_by_model": usage, "audit_csv": str(out), "audit_rows": len(picked)}
    (repo_path("runs") / "judge_all_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
