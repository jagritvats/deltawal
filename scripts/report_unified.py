"""Unified LongMemEval-S held-out table (138 questions, one judge): baselines + DeltaWAL-Versioned at lag 0/1/3/5.

Rows per reader (all p_ooo=0):
  B1_sync (mutable, lag0) | B2_async (lag1) | B3_recency_matched_user (lag1) | deltawal_shadow_v2 (lag1) |
  deltawal_lazy (lag1) | deltawal_prompt (mutable, lag1) | DeltaWAL-Versioned (lag 0, 1, 3, 5)
Every cell is measured natively (DeltaWAL-Versioned lag 0 included, for all three readers). The B1_sync_versioned
stand-in (same stored state, empty WAL, different tag wording) is only used -- and footnoted with a dagger -- if a
native lag-0 cell is missing.
"""
from __future__ import annotations

import json
import sys

from epochmem.config import repo_path

sys.path.insert(0, str(repo_path("scripts")))
from report_heldout import boot_ci  # noqa: E402

READERS = [("gemma4", "m2ho_gemma", "m2ctl_gemma", "m2uni_gemma"),
           ("glm-5.3", "m2ho_glm", "m2ctl_glm", "m2uni_glm"),
           ("gpt-6-luna", "m2ho_luna6", "m2ctl_luna6", "m2uni_luna6")]
STRATA = [("All", lambda r: True), ("KU", lambda r: r["qtype"] == "knowledge-update" and not r["is_abstention"]),
          ("TR", lambda r: r["qtype"] == "temporal-reasoning" and not r["is_abstention"]), ("Abs", lambda r: r["is_abstention"])]


def load(run):
    p = repo_path("runs") / run / "results_judged.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


def cell(rows, system, lag):
    return [r for r in rows if r["system"] == system and r["lag"] == lag and r["p_ooo"] == 0.0]


def fmt(rs, f):
    xs = [float(r["correct"]) for r in rs if f(r)]
    if not xs:
        return "-"
    m, lo, hi = boot_ci(xs)
    return f"{100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]"


def main() -> int:
    lines, stand_in = [], False
    for name, main_run, ctl_run, uni_run in READERS:
        M, C, U = load(main_run), load(ctl_run), load(uni_run)
        spec = [("B1_sync (mutable)", 0, cell(M, "B1_sync", 0)),
                ("B2_async", 1, cell(M, "B2_async", 1)),
                ("B3_recency_matched_user", 1, cell(M, "B3_recency_matched_user", 1)),
                ("deltawal_shadow_v2", 1, cell(M, "deltawal_shadow_v2", 1)),
                ("deltawal_lazy", 1, cell(M, "deltawal_lazy", 1)),
                ("deltawal_prompt (mutable)", 1, cell(M, "deltawal_prompt", 1))]
        v0 = cell(U, "deltawal_versioned", 0)
        stand_in |= not v0
        spec.append(("DeltaWAL-Versioned" if v0 else "DeltaWAL-Versioned (B1_sync_versioned)†", 0,
                     v0 or cell(C, "B1_sync_versioned", 0)))
        for lag in (1, 3, 5):
            rs = cell(U, "deltawal_versioned", lag)
            if rs:
                spec.append(("DeltaWAL-Versioned", lag, rs))
        lines.append(f"\n=== reader={name}")
        lines.append(f"{'system':<42}{'lag':>4} | " + " | ".join(f"{s:^20}" for s, _ in STRATA) + f" | {'SRR-KU':>6} {'tokens':>6}")
        for sname, lag, rs in spec:
            ku = [r for r in rs if STRATA[1][1](r)]
            srr = 100 * sum((not r["correct"]) and r["uses_superseded"] for r in ku) / len(ku) if ku else float("nan")
            tok = sum(r["prompt_tokens"] for r in rs) / len(rs)
            lines.append(f"{sname:<42}{lag:>4} | " + " | ".join(f"{fmt(rs, f):^20}" for _, f in STRATA) + f" | {srr:6.1f} {tok:6.0f}")
        if name == "gemma4":
            a, b = cell(U, "deltawal_versioned", 0), cell(C, "B1_sync_versioned", 0)
            if a and b:
                A = {r["qid"]: r["correct"] for r in a}
                B = {r["qid"]: r["correct"] for r in b}
                agree = sum(A[q] == B[q] for q in A) / len(A)
                lines.append(f"  [proxy check] gemma lag0: deltawal_versioned {100 * sum(A.values()) / len(A):.1f} vs "
                             f"B1_sync_versioned {100 * sum(B.values()) / len(B):.1f}; per-question verdict agreement {100 * agree:.1f}%")
    if stand_in:
        lines.append("\n† lag-0 DeltaWAL-Versioned for cloud readers = B1_sync_versioned (same stored state, empty WAL; tag wording differs).")
    out = "\n".join(lines)
    print(out)
    (repo_path("runs") / "report_unified.txt").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
