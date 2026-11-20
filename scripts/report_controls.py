"""Side-by-side control table on the 138 held-out questions (all judged by the same judge):

  B1_sync (mutable, lag 0) | deltawal_prompt (mutable, lag 0) | B1_sync_versioned (lag 0) | deltawal_prompt (lag 1)

Accuracy with 95% bootstrap CIs (resampling questions), split All / KU / KU-past / KU-current / TR / Abs,
plus how the 13 'destructive overwrite' history cases from forensics #1 fare under each system.
"""
from __future__ import annotations

import json
import re
import sys

from epochmem.config import repo_path

sys.path.insert(0, str(repo_path("scripts")))
from report_heldout import boot_ci  # noqa: E402

READERS = [("gemma4", "m2ho_gemma", "m2ctl_gemma"), ("glm-5.3", "m2ho_glm", "m2ctl_glm"),
           ("gpt-6-luna", "m2ho_luna6", "m2ctl_luna6")]
COLS = [("B1_sync (mutable, lag0)", "main", "B1_sync", 0),
        ("deltawal_prompt (mutable, lag0)", "ctl", "deltawal_prompt", 0),
        ("B1_sync_versioned (lag0)", "ctl", "B1_sync_versioned", 0),
        ("deltawal_prompt (lag1)", "main", "deltawal_prompt", 1)]
PAST = re.compile(r"\b(initial|initially|originally|first|before|previous|previously|used to|at first|earlier|prior|"
                  r"than i did|than before|switch|switched)\b", re.I)


def load(run):
    return [json.loads(l) for l in open(repo_path("runs") / run / "results_judged.jsonl", encoding="utf-8")]


def main() -> int:
    fx = json.loads((repo_path("runs") / "forensics_heldout.json").read_text(encoding="utf-8"))
    lines = []
    strata = [("All", lambda r: True),
              ("KU", lambda r: r["qtype"] == "knowledge-update" and not r["is_abstention"]),
              ("KU-past/comparative", lambda r: r["qtype"] == "knowledge-update" and not r["is_abstention"] and bool(PAST.search(r["question"]))),
              ("KU-current", lambda r: r["qtype"] == "knowledge-update" and not r["is_abstention"] and not PAST.search(r["question"])),
              ("TR", lambda r: r["qtype"] == "temporal-reasoning" and not r["is_abstention"]),
              ("Abs", lambda r: r["is_abstention"])]
    for name, main_run, ctl_run in READERS:
        src = {"main": load(main_run), "ctl": load(ctl_run)}
        lines.append(f"\n=== reader={name}")
        lines.append(f"{'stratum':<22}{'n':>4} | " + " | ".join(f"{c[0]:^31}" for c in COLS))
        for sname, f in strata:
            cells, n = [], None
            for _, which, system, lag in COLS:
                rs = [r for r in src[which] if r["system"] == system and r["lag"] == lag and r["p_ooo"] == 0.0 and f(r)]
                n = len(rs)
                m, lo, hi = boot_ci([float(r["correct"]) for r in rs])
                cells.append(f"{100 * m:5.1f} [{100 * lo:4.1f},{100 * hi:5.1f}]" if rs else "-")
            lines.append(f"{sname:<22}{n:>4} | " + " | ".join(f"{c:^31}" for c in cells))
        # stale-read rate on KU and mean prompt tokens
        sr, tk = [], []
        for _, which, system, lag in COLS:
            rs = [r for r in src[which] if r["system"] == system and r["lag"] == lag and r["p_ooo"] == 0.0]
            ku = [r for r in rs if strata[1][1](r)]
            sr.append(f"{100 * sum((not r['correct']) and r['uses_superseded'] for r in ku) / len(ku):5.1f}")
            tk.append(f"{sum(r['prompt_tokens'] for r in rs) / len(rs):6.0f}")
        lines.append(f"{'stale-read rate (KU)':<26} | " + " | ".join(f"{x:^31}" for x in sr))
        lines.append(f"{'mean prompt tokens':<26} | " + " | ".join(f"{x:^31}" for x in tk))
    # the 13 destructive-overwrite history cases from forensics #1 (glm-5.3 + gpt-6-luna)
    lines.append("\n=== Forensics #1 '(a) destructive overwrite' cases: correct under each system")
    for name, main_run, ctl_run in READERS[1:]:
        src = {"main": load(main_run), "ctl": load(ctl_run)}
        q1 = [l for l in (repo_path("runs") / "forensics_heldout.txt").read_text(encoding="utf-8", errors="replace").splitlines()
              if l.strip().startswith(main_run) and "(a)" in l]
        qids = [l.split()[1] for l in q1]
        row = []
        for cname, which, system, lag in COLS:
            ok = sum(1 for r in src[which] if r["system"] == system and r["lag"] == lag and r["p_ooo"] == 0.0
                     and r["qid"] in qids and r["correct"])
            row.append(f"{cname}: {ok}/{len(qids)}")
        lines.append(f"{name:<12} " + " | ".join(row))
    out = "\n".join(lines)
    print(out)
    (repo_path("runs") / "report_controls.txt").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
