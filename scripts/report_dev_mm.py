"""3-model dev comparison on the core cells (all readers judged by the same cloud judge)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from epochmem.config import repo_path

SYSTEMS = ["B1_sync", "B2_async", "B3_recency", "deltawal_prompt", "deltawal_shadow_v2", "deltawal_lazy"]
CELLS = [(0, 0.0), (1, 0.0), (3, 0.0), (1, 0.3)]


def pct(xs):
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):5.1f}" if xs else "   - "


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run ids, one per reader")
    args = ap.parse_args()
    lines = []
    for run in args.runs:
        rows = [json.loads(l) for l in open(repo_path("runs") / run / "results.jsonl", encoding="utf-8")]
        reader = rows[0].get("reader_model", "?")
        judge = rows[0].get("judge_model", "?")
        lines.append(f"\n=== reader={reader}  judge={judge}  ({run}, {len(rows)} rows)")
        lines.append(f"{'system':<20}{'lag':>4}{'p_ooo':>6} | {'Acc':>6}{'KU':>7}{'Abs':>7}{'TR':>7} | {'CWR':>6}"
                     f"{'SRR-KU':>8} | {'OOCR':>6}{'MissM':>7} | {'tokens':>7}{'#sup':>6}")
        cell = defaultdict(list)
        for r in rows:
            cell[(r["system"], r["lag"], r["p_ooo"])].append(r)
        for s in SYSTEMS:
            for lag, p in CELLS:
                rs = cell.get((s, lag, p))
                if not rs:
                    continue
                ku = [r for r in rs if r["qtype"] == "knowledge-update" and not r["is_abstention"]]
                up = sum(r["oocr_updated"] for r in rs)
                co = sum(r["oocr_corrupted"] for r in rs)
                mm = [r["missed_merge"] for r in ku if r.get("missed_merge") is not None]
                lines.append(
                    f"{s:<20}{lag:>4}{p:>6.2f} | {pct(r['correct'] for r in rs):>6}{pct(r['correct'] for r in ku):>7}"
                    f"{pct(r['correct'] for r in rs if r['is_abstention']):>7}"
                    f"{pct(r['correct'] for r in rs if r['qtype'] == 'temporal-reasoning' and not r['is_abstention']):>7} | "
                    f"{pct((not r['correct']) and not r['abstained'] for r in rs):>6}"
                    f"{pct((not r['correct']) and r['uses_superseded'] for r in ku):>8} | "
                    f"{(100 * co / up if up else float('nan')):>6.1f}{pct(mm):>7} | "
                    f"{sum(r['prompt_tokens'] for r in rs) / len(rs):>7.0f}"
                    f"{sum(r.get('n_suppressed', 0) for r in rs) / len(rs):>6.2f}")
    out = "\n".join(lines)
    print(out)
    (repo_path("runs") / "report_dev_mm.txt").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
