"""Held-out report: accuracy with 95% bootstrap CIs (resampling questions) per reader/system/cell,
split by stratum; stale-read and confident-wrong rates; missed-merge rescue counts (shadow_v2 vs
lazy at p_ooo=0.3); cloud token spend per model."""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict

from epochmem.config import repo_path

SYSTEMS = ["B1_sync", "B2_async", "B3_recency_matched_user", "deltawal_prompt", "deltawal_shadow_v2", "deltawal_lazy"]
STRATA = [("all", lambda r: True), ("KU", lambda r: r["qtype"] == "knowledge-update" and not r["is_abstention"]),
          ("TR", lambda r: r["qtype"] == "temporal-reasoning" and not r["is_abstention"]),
          ("Abs", lambda r: r["is_abstention"])]


def boot_ci(vals: list[float], iters: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    if not vals:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(vals) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def fmt(t):
    m, lo, hi = t
    return "   -          " if m != m else f"{100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="runs/report_heldout.txt")
    args = ap.parse_args()
    lines, rescue_lines = [], []
    for run in args.runs:
        rows = [json.loads(l) for l in open(repo_path("runs") / run / "results_judged.jsonl", encoding="utf-8")]
        reader = rows[0]["reader_model"]
        lines.append(f"\n=== reader={reader}  judge={rows[0]['judge_model']}  run={run}  rows={len(rows)}  "
                     f"questions={len({r['qid'] for r in rows})}")
        lines.append(f"{'system':<24}{'lag':>4}{'p':>5} | " + " | ".join(f"{s:^20}" for s, _ in STRATA)
                     + f" | {'CWR':>5} {'SRR-KU':>6} | {'tokens':>6} {'#sup':>5}")
        cell = defaultdict(list)
        for r in rows:
            cell[(r["system"], r["lag"], r["p_ooo"])].append(r)
        for s in SYSTEMS:
            for (sys_, lag, p) in sorted(k for k in cell if k[0] == s):
                rs = cell[(sys_, lag, p)]
                accs = [fmt(boot_ci([float(r["correct"]) for r in rs if f(r)])) for _, f in STRATA]
                ku = [r for r in rs if STRATA[1][1](r)]
                cwr = sum((not r["correct"]) and not r["abstained"] for r in rs) / len(rs)
                srr = sum((not r["correct"]) and r["uses_superseded"] for r in ku) / len(ku) if ku else float("nan")
                lines.append(f"{s:<24}{lag:>4}{p:>5.2f} | " + " | ".join(f"{a:^20}" for a in accs)
                             + f" | {100 * cwr:5.1f} {100 * srr:6.1f} | {sum(r['prompt_tokens'] for r in rs) / len(rs):>6.0f}"
                             f" {sum(r.get('n_suppressed', 0) for r in rs) / len(rs):>5.2f}")
        # missed-merge rescue: shadow_v2 correct, lazy wrong, lazy failed by key drift (missed merge)
        for (lag, p) in sorted({(k[1], k[2]) for k in cell if k[2] > 0}):
            sh = {r["qid"]: r for r in cell.get(("deltawal_shadow_v2", lag, p), [])}
            lz = {r["qid"]: r for r in cell.get(("deltawal_lazy", lag, p), [])}
            both = sh.keys() & lz.keys()
            rescue = sorted(q for q in both if sh[q]["correct"] and not lz[q]["correct"] and lz[q].get("missed_merge"))
            sh_only = sorted(q for q in both if sh[q]["correct"] and not lz[q]["correct"])
            lz_only = sorted(q for q in both if lz[q]["correct"] and not sh[q]["correct"])
            mm = sum(1 for q in both if lz[q].get("missed_merge"))
            rescue_lines.append(f"{reader:<28} lag={lag} p_ooo={p}: missed-merge rescues={len(rescue)} {rescue} | "
                                f"shadow-only correct={len(sh_only)} lazy-only correct={len(lz_only)} | "
                                f"questions with missed merge={mm}/{len(both)}")
    lines.append("\n=== Missed-merge rescue (shadow_v2 correct, lazy wrong, update split across keys)")
    lines += rescue_lines
    # token spend per model
    spend = defaultdict(lambda: {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})
    for f in glob.glob(str(repo_path("runs") / "*" / "stats.json")) + [str(repo_path("runs/judge_all_summary.json"))]:
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        if "heldout" not in f and "judge_all" not in f and "m2ho_" not in f:
            continue
        for m, u in (d.get("cloud_usage_by_model") or d.get("usage_by_model") or {}).items():
            for k in spend[m]:
                spend[m][k] += u.get(k, 0)
    for f in glob.glob(str(repo_path("runs") / "*heldout*_usage.json")):
        for m, u in json.load(open(f)).items():
            for k in spend[m]:
                spend[m][k] += u.get(k, 0)
    lines.append("\n=== Cloud token spend (live calls only; cached calls cost nothing)")
    for m, u in sorted(spend.items()):
        lines.append(f"{m:<24} calls={u['calls']:>6}  prompt={u['prompt_tokens']:>11,}  completion={u['completion_tokens']:>10,}")
    out = "\n".join(lines)
    print(out)
    (repo_path(args.out)).write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
