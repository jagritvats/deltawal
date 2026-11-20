"""Post-hoc shadow_v2 threshold sensitivity on held-out gemma cells.

Uses the raw per-fact max newer-sentence similarities saved at run time (`shadow_scores`). For each
threshold t, a cell's masked set is {facts with score >= t}. Cells whose masked set equals the
frozen-threshold set reuse the frozen verdict; only cells whose set changes are re-run (reader
only, same cached extraction/embeddings) and judged with the same deduplicated judge.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from epochmem.config import load_config, repo_path
from epochmem.guard import set_allow_cloud
from epochmem.runner import build_ctx, enforce_provider_caps, select_instances
from epochmem.systems import REGISTRY

import sys
sys.path.insert(0, str(repo_path("scripts")))
from judge_all import judge_pairs, make_judge, normalize_answer  # noqa: E402
from report_heldout import boot_ci  # noqa: E402

THRESHOLDS = [0.60, 0.65, 0.70, 0.72, 0.75]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_heldout.yaml")
    ap.add_argument("--run-id", required=True, help="judged gemma held-out run")
    ap.add_argument("--judge", default="opencode-go/qwen4-max")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    ap.add_argument("--allow-go-scale", action="store_true", default=False)
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    cfg.setdefault("runtime", {})["allow_go_scale"] = args.allow_go_scale
    frozen = cfg["systems"]["shadow_v2_threshold"]
    rows = [json.loads(l) for l in open(repo_path("runs") / args.run_id / "results_judged.jsonl", encoding="utf-8")]
    rows = [r for r in rows if r["system"] == "deltawal_shadow_v2"]
    preps = {p.inst.qid: p for p in select_instances(cfg, None)}
    enforce_provider_caps(cfg, len(preps))
    ctx, _ = build_ctx(cfg, session_id="ses_epochmem_sensitivity")
    judge, clouds = make_judge(cfg, args.judge, "ses_epochmem_sensitivity_judge")

    results = defaultdict(dict)        # t -> row_key -> verdict-bearing dict
    reruns = defaultdict(list)
    for r in rows:
        scores = r["read_diag"].get("shadow_scores", [])
        frozen_set = {(k, e) for k, e, s in scores if s >= frozen}
        for t in THRESHOLDS:
            tset = {(k, e) for k, e, s in scores if s >= t}
            if tset == frozen_set:
                results[t][r["row_key"]] = r
            else:
                reruns[t].append(r)
    print({t: len(v) for t, v in reruns.items()}, "cells re-run per threshold", flush=True)
    new_answers = {}
    from concurrent.futures import ThreadPoolExecutor

    def rerun(t, r):
        sysobj = REGISTRY["deltawal_shadow_v2"](ctx)
        ns = f"sens:{t}:{r['row_key']}"
        out = sysobj.run(preps[r["qid"]], ns, r["lag"], r["p_ooo"])
        ctx.store.reset(ns)
        ctx.wal.reset(ns)
        return (t, r["row_key"]), (r, out.answer)

    for t, rs in sorted(reruns.items()):          # one threshold at a time (threshold lives in cfg)
        cfg["systems"]["shadow_v2_threshold"] = t
        with ThreadPoolExecutor(args.workers) as ex:
            for k, v in ex.map(lambda r: rerun(t, r), rs):
                new_answers[k] = v
        print(f"t={t}: re-ran {len(rs)} cells", flush=True)
    cfg["systems"]["shadow_v2_threshold"] = frozen
    pairs = {}
    for (t, key), (r, ans) in new_answers.items():
        pairs.setdefault((r["qid"], normalize_answer(ans)), ans)
    verdicts = judge_pairs(pairs, preps, judge, 16) if pairs else {}
    for (t, key), (r, ans) in new_answers.items():
        results[t][key] = {**r, "answer": ans, **verdicts[(r["qid"], normalize_answer(ans))]}

    lines = [f"shadow_v2 threshold sensitivity on {args.run_id} (frozen={frozen}); cells={len(rows)}",
             f"{'t':>5} {'reruns':>7} | {'all':^20} | {'KU':^20} | {'TR':^20} | {'Abs':^20}"]
    strata = [("all", lambda r: True), ("KU", lambda r: r["qtype"] == "knowledge-update" and not r["is_abstention"]),
              ("TR", lambda r: r["qtype"] == "temporal-reasoning" and not r["is_abstention"]),
              ("Abs", lambda r: r["is_abstention"])]
    for t in THRESHOLDS:
        vals = list(results[t].values())
        cis = []
        for _, f in strata:
            m, lo, hi = boot_ci([float(v["correct"]) for v in vals if f(v)])
            cis.append(f"{100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]")
        lines.append(f"{t:>5.2f} {len(reruns.get(t, [])):>7} | " + " | ".join(f"{c:^20}" for c in cis))
    usage = {m: u for c in list(clouds.values()) + list(ctx.client.clouds.values()) for m, u in c.usage_by_model.items()}
    lines.append(f"cloud usage: {json.dumps(usage)}")
    out = "\n".join(lines)
    print(out)
    (repo_path("runs") / "sensitivity_shadow_heldout.txt").write_text(out, encoding="utf-8")
    (repo_path("runs") / "sensitivity_heldout_usage.json").write_text(json.dumps(usage), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
