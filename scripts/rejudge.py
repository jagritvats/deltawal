"""Re-judge an existing run's answers with a different judge model and report agreement.

Only judge calls are made (no reader calls). Rows are deduplicated by (qid, answer) since the
judge verdict depends only on those. Reports exact agreement and Cohen's kappa per verdict field.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from epochmem.cache import DiskCache
from epochmem.config import load_config, make_cloud_client, repo_path
from epochmem.data.stream import prepare
from epochmem.eval.judge import Judge
from epochmem.guard import set_allow_cloud
from epochmem.llm import CLOUD_PREFIX, ModelRouter, OllamaClient
from epochmem.runner import select_instances


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_dev_local.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--judge", default="opencode-go/qwen4-max")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    cache = DiskCache(repo_path(cfg["runtime"]["cache_path"]))
    m = cfg["models"]
    local = OllamaClient(m["endpoint"], cache, num_ctx=m["num_ctx"], seed=cfg["seed"], think=m["think"])
    cloud = make_cloud_client(cfg, cache, session_id=f"ses_epochmem_rejudge_{args.run_id}") \
        if args.judge.startswith(CLOUD_PREFIX) else None
    judge = Judge(ModelRouter(local, cloud, (cfg.get("cloud") or {}).get("min_max_tokens", 2048)), args.judge)

    run_dir = repo_path("runs") / args.run_id
    rows = [json.loads(l) for l in open(run_dir / "results.jsonl", encoding="utf-8")]
    preps = {p.inst.qid: p for p in select_instances(cfg, None)}
    pairs = sorted({(r["qid"], r["answer"]) for r in rows})
    print(f"{len(rows)} rows -> {len(pairs)} unique (qid, answer) pairs; judge={args.judge}", flush=True)

    def work(pair):
        qid, ans = pair
        v = judge.judge(preps[qid], ans)
        return pair, v

    with ThreadPoolExecutor(args.workers) as ex:
        verdicts = dict(ex.map(work, pairs))

    out_rows, fields = [], ("correct", "abstained", "uses_superseded")
    agree = {f: ([], []) for f in fields}
    for r in rows:
        v = verdicts[(r["qid"], r["answer"])]
        new = {"correct": v.correct, "abstained": v.abstained, "uses_superseded": v.uses_superseded}
        for f in fields:
            if f == "uses_superseded" and not (r["qtype"] == "knowledge-update" and not r["is_abstention"]):
                continue
            agree[f][0].append(bool(r[f]))
            agree[f][1].append(new[f])
        out_rows.append({"row_key": r["row_key"], "local": {f: r[f] for f in fields}, "cloud": new,
                         "cloud_parse_errors": v.judge_parse_errors})
    tag = args.judge.replace("/", "_")
    with open(run_dir / f"rejudge_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for o in out_rows:
            fh.write(json.dumps(o) + "\n")
    summary = {"judge": args.judge, "rows": len(rows), "unique_pairs": len(pairs),
               "cloud_parse_errors": sum(v.judge_parse_errors for v in verdicts.values()),
               "cloud_usage": cloud.usage if cloud else None}
    for f in fields:
        a, b = agree[f]
        summary[f] = {"n": len(a), "agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 4) if a else None,
                      "kappa": round(kappa(a, b), 4), "local_pos_rate": round(sum(a) / len(a), 4) if a else None,
                      "cloud_pos_rate": round(sum(b) / len(b), 4) if b else None,
                      "local_true_cloud_false": sum(x and not y for x, y in zip(a, b)),
                      "local_false_cloud_true": sum(y and not x for x, y in zip(a, b))}
    # same on unique pairs (each verdict counted once)
    lp = {}
    for r in rows:
        lp[(r["qid"], r["answer"])] = bool(r["correct"])
    a = [lp[p] for p in pairs]
    b = [verdicts[p].correct for p in pairs]
    summary["correct_unique_pairs"] = {"n": len(a), "agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 4),
                                       "kappa": round(kappa(a, b), 4)}
    (run_dir / f"rejudge_{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
