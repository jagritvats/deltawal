"""External-validation tables with 95% bootstrap CIs (resampling questions), one judge throughout.

  --stage regression   : 60 non-update LongMemEval-S questions, split by question type
  --stage mem0         : Mem0 OSS on the 60 held-out KU questions (KU / KU-past / KU-current, stale reads,
                         Mem0 history ADD/UPDATE/DELETE counts on KU-past questions)
  --stage memconflict  : 90 MemConflict questions, split by conflict type, with trap rates
                         (dynamic: answered the superseded value; static: believed the newer FALSE statement)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter

from epochmem.config import repo_path

sys.path.insert(0, str(repo_path("scripts")))
from report_heldout import boot_ci  # noqa: E402

PAST = re.compile(r"\b(initial|initially|originally|first|before|previous|previously|used to|at first|earlier|prior|"
                  r"than i did|than before|switch|switched)\b", re.I)

STAGES = {
    "regression": {
        "readers": [("gemma4", "m3r60_gemma"), ("gpt-6-luna", "m3r60_luna6"), ("glm-5.3", "m3r60_glm")],
        "cells": [("B1_sync (mutable)", "B1_sync", 0), ("DeltaWAL-Versioned", "deltawal_versioned", 0),
                  ("B2_async", "B2_async", 1), ("B3_recency_matched_user", "B3_recency_matched_user", 1),
                  ("DeltaWAL-Versioned", "deltawal_versioned", 1)],
        "strata": [("All", lambda r: True), ("multi-session", lambda r: r["qtype"] == "multi-session"),
                   ("ss-user", lambda r: r["qtype"] == "single-session-user"),
                   ("ss-assistant", lambda r: r["qtype"] == "single-session-assistant"),
                   ("ss-preference", lambda r: r["qtype"] == "single-session-preference")],
    },
    "mem0": {
        "readers": [("gemma4", "m3mem0_gemma"), ("gpt-6-luna", "m3mem0_luna6"), ("glm-5.3", "m3mem0_glm")],
        "cells": [("Mem0_sync", "Mem0_sync", 0), ("Mem0_async", "Mem0_async", 1),
                  ("Mem0 + DeltaWAL wrapper", "Mem0_deltawal", 1)],
        "strata": [("KU", lambda r: True), ("KU-past", lambda r: bool(PAST.search(r["question"]))),
                   ("KU-current", lambda r: not PAST.search(r["question"]))],
    },
    "memconflict": {
        "readers": [("gemma4", "m3mc_gemma"), ("gpt-6-luna", "m3mc_luna6"), ("glm-5.3-flash", "m3mc_glmflash")],
        "cells": [("B1_sync (mutable)", "B1_sync", 0), ("DeltaWAL-Versioned", "deltawal_versioned", 0),
                  ("B2_async", "B2_async", 1), ("B3_recency_matched_user", "B3_recency_matched_user", 1),
                  ("DeltaWAL-Versioned", "deltawal_versioned", 1)],
        "strata": [("All", lambda r: True), ("dynamic", lambda r: r["qtype"] == "memconflict-dynamic"),
                   ("static", lambda r: r["qtype"] == "memconflict-static"),
                   ("conditional", lambda r: r["qtype"] == "memconflict-conditional")],
    },
}


def fmt(xs):
    if not xs:
        return "-"
    m, lo, hi = boot_ci(xs)
    return f"{100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    args = ap.parse_args()
    st = STAGES[args.stage]
    lines = []
    for rname, run in st["readers"]:
        p = repo_path("runs") / run / "results_judged.jsonl"
        if not p.exists():
            lines.append(f"\n=== reader={rname}: (no judged results at {p})")
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        lines.append(f"\n=== reader={rname}  ({run}, {len(rows)} rows, {len({r['qid'] for r in rows})} questions)")
        hdr = f"{'system':<26}{'lag':>4} | " + " | ".join(f"{s:^20}" for s, _ in st["strata"])
        extra = " | trap/stale  tokens" if args.stage != "regression" else " | tokens"
        lines.append(hdr + extra)
        for cname, system, lag in st["cells"]:
            rs = [r for r in rows if r["system"] == system and r["lag"] == lag and r.get("p_ooo", 0.0) == 0.0]
            if not rs:
                continue
            cells = [fmt([float(r["correct"]) for r in rs if f(r)]) for _, f in st["strata"]]
            tok = sum(r["prompt_tokens"] for r in rs) / len(rs)
            if args.stage == "regression":
                tail = f" | {tok:6.0f}"
            else:
                if args.stage == "mem0":
                    pool = rs
                else:
                    pool = [r for r in rs if r["qtype"] in ("memconflict-dynamic", "memconflict-static")]
                trap = 100 * sum((not r["correct"]) and r["uses_superseded"] for r in pool) / len(pool) if pool else float("nan")
                tail = f" | {trap:9.1f}  {tok:6.0f}"
                if args.stage == "memconflict":
                    by = {t: [r for r in rs if r["qtype"] == f"memconflict-{t}"] for t in ("dynamic", "static")}
                    tail += "  (" + ", ".join(
                        f"{t} trap {100 * sum((not r['correct']) and r['uses_superseded'] for r in v) / len(v):.1f}"
                        for t, v in by.items() if v) + ")"
            lines.append(f"{cname:<26}{lag:>4} | " + " | ".join(f"{c:^20}" for c in cells) + tail)
    if args.stage == "mem0":
        kp = repo_path("runs/m3mem0_luna6/mem0_ku_past.jsonl")
        if kp.exists():
            recs = [json.loads(l) for l in open(kp, encoding="utf-8")]
            agg = Counter()
            for r in recs:
                for k, v in (r.get("totals_B") or {}).items():
                    if isinstance(v, (int, float)):
                        agg[k] += v
            lines.append(f"\n=== Mem0 history on KU-past questions ({len(recs)} questions): {dict(agg)}")
            keys = sorted({k for r in recs for k, v in r.items() if isinstance(v, bool)})
            for k in keys:
                lines.append(f"    {k}: {sum(1 for r in recs if r.get(k))}/{len(recs)}")
    out = "\n".join(lines)
    print(out)
    (repo_path("runs") / f"report_m3_{args.stage}.txt").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
