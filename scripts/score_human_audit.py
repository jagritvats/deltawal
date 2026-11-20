"""Score the human audit: human labels vs. the automatic judge (qwen3.8-flash) on runs/human_audit_50.csv.

Fill the `human_correct`, `human_abstained`, `human_uses_superseded` columns with 1/0, true/false, yes/no or y/n.
Blank cells are skipped for that field (n is reported per field). Reports exact agreement, Cohen's kappa and
95% bootstrap CIs (resampling audit rows), plus the confusion counts, for:
  Correct, Abstained, and Stale-Read = (not correct) and uses_superseded  [knowledge-update rows only].
"""
from __future__ import annotations

import argparse
import csv
import json
import random

from epochmem.config import repo_path

TRUE = {"1", "true", "t", "yes", "y"}
FALSE = {"0", "false", "f", "no", "n"}


def parse(v: str | None) -> bool | None:
    v = (v or "").strip().lower()
    return True if v in TRUE else False if v in FALSE else None


def kappa(a: list[bool], b: list[bool]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def stats(a: list[bool], b: list[bool], iters: int = 2000, seed: int = 42) -> dict:
    n = len(a)
    if n == 0:
        return {"n": 0}
    rng = random.Random(seed)
    ag, ks = [], []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        aa, bb = [a[i] for i in idx], [b[i] for i in idx]
        ag.append(sum(x == y for x, y in zip(aa, bb)) / n)
        ks.append(kappa(aa, bb))
    ag.sort()
    ks.sort()
    lo, hi = int(0.025 * iters), int(0.975 * iters) - 1
    return {"n": n, "agreement": round(sum(x == y for x, y in zip(a, b)) / n, 4),
            "agreement_ci95": [round(ag[lo], 4), round(ag[hi], 4)],
            "kappa": round(kappa(a, b), 4), "kappa_ci95": [round(ks[lo], 4), round(ks[hi], 4)],
            "human_true_judge_true": sum(x and y for x, y in zip(a, b)),
            "human_true_judge_false": sum(x and not y for x, y in zip(a, b)),
            "human_false_judge_true": sum(y and not x for x, y in zip(a, b)),
            "human_false_judge_false": sum((not x) and (not y) for x, y in zip(a, b))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/human_audit_50.csv")
    ap.add_argument("--out", default="runs/human_audit_scores.json")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(repo_path(args.csv), encoding="utf-8")))
    fields = {"correct": [], "abstained": [], "stale_read": []}
    for r in rows:
        hc, jc = parse(r["human_correct"]), parse(r["judge_correct"])
        ha, ja = parse(r["human_abstained"]), parse(r["judge_abstained"])
        hs, js = parse(r["human_uses_superseded"]), parse(r["judge_uses_superseded"])
        if hc is not None and jc is not None:
            fields["correct"].append((hc, jc))
        if ha is not None and ja is not None:
            fields["abstained"].append((ha, ja))
        if r["stratum"] == "knowledge-update" and None not in (hc, hs, jc, js):
            fields["stale_read"].append(((not hc) and hs, (not jc) and js))
    labeled = sum(1 for r in rows if parse(r["human_correct"]) is not None)
    if labeled == 0:
        print(f"No human labels found in {args.csv} ({len(rows)} rows). Fill human_correct / human_abstained / "
              f"human_uses_superseded with 1/0 and re-run.")
        return 1
    out = {"csv": args.csv, "rows": len(rows), "rows_with_human_correct": labeled,
           **{k: stats([h for h, _ in v], [j for _, j in v]) for k, v in fields.items()}}
    (repo_path(args.out)).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
