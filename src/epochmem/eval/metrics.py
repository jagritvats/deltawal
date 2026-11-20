"""Aggregate per-row results into the Milestone 1 table."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

SYSTEM_ORDER = ["B1_sync", "B1_sync_versioned", "B2_async", "B3_recency", "B3_recency_matched", "B3_recency_matched_user",
                "B4_epoch_wal", "deltawal_prompt", "deltawal_shadow",
                "deltawal_shadow_v2", "deltawal_lazy", "deltawal_versioned"]


def load_rows(path: Path) -> list[dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["row_key"]] = r   # last write wins on accidental duplicates
    return list(rows.values())


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(rows: list[dict]) -> list[dict]:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        cells[(r["system"], r["lag"], r["p_ooo"])].append(r)
    out = []
    for (system, lag, p), rs in cells.items():
        ku = [r for r in rs if r["qtype"] == "knowledge-update" and not r["is_abstention"]]
        ku_inflight = [r for r in ku if r["evidence_in_flight"]]
        upd = sum(r["oocr_updated"] for r in rs)
        cor = sum(r["oocr_corrupted"] for r in rs)
        out.append({
            "system": system, "lag": lag, "p_ooo": p, "n": len(rs),
            "acc": _mean(r["correct"] for r in rs),
            "acc_ku": _mean(r["correct"] for r in ku),
            "acc_tr": _mean(r["correct"] for r in rs if r["qtype"] == "temporal-reasoning" and not r["is_abstention"]),
            "acc_abs": _mean(r["correct"] for r in rs if r["is_abstention"]),
            "cwr": _mean((not r["correct"]) and (not r["abstained"]) for r in rs),
            "srr_ku": _mean((not r["correct"]) and r["uses_superseded"] for r in ku),
            "srr_ku_inflight": _mean((not r["correct"]) and r["uses_superseded"] for r in ku_inflight),
            "n_ku_inflight": len(ku_inflight),
            "oocr": (cor / upd) if upd else float("nan"),
            "oocr_updated": upd,
            "tokens": _mean(r["prompt_tokens"] for r in rs),
            "read_ms": _median([r.get("read_path_ms", 0.0) + r.get("read_llm_ms", 0.0) for r in rs]),
            "suppressed": _mean(r.get("n_suppressed", 0) for r in rs),
            "missed_merge": _mean(r["missed_merge"] for r in ku if r.get("missed_merge") is not None),
        })
    out.sort(key=lambda d: (SYSTEM_ORDER.index(d["system"]) if d["system"] in SYSTEM_ORDER else 99, d["p_ooo"], d["lag"]))
    return out


def _median(xs):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def _f(x, pct=True):
    if x != x:  # nan
        return "   -  "
    return f"{100 * x:5.1f}%" if pct else f"{x:6.0f}"


def format_table(agg: list[dict]) -> str:
    hdr = (f"{'system':<14}{'p_ooo':>6}{'lag':>5}{'n':>4} | {'Acc':>6}{'Acc-KU':>8}{'Acc-TR':>8}{'Acc-Abs':>8} | "
           f"{'CWR':>6}{'SRR-KU':>8}{'SRR-KU*':>8} | {'OOCR':>6}{'#upd':>6} | {'tokens':>7}{'readms':>8}{'#sup':>6}{'MissM':>7}")
    lines = [hdr, "-" * len(hdr)]
    prev = None
    for d in agg:
        if prev is not None and (d["system"], d["p_ooo"]) != prev:
            lines.append("")
        prev = (d["system"], d["p_ooo"])
        lines.append(
            f"{d['system']:<14}{d['p_ooo']:>6.2f}{d['lag']:>5}{d['n']:>4} | {_f(d['acc']):>6}{_f(d['acc_ku']):>8}"
            f"{_f(d['acc_tr']):>8}{_f(d['acc_abs']):>8} | {_f(d['cwr']):>6}{_f(d['srr_ku']):>8}"
            f"{_f(d['srr_ku_inflight']):>8} | {_f(d['oocr']):>6}{d['oocr_updated']:>6} | {_f(d['tokens'], False):>7}{d['read_ms']:>8.1f}{d['suppressed']:>6.2f}{_f(d['missed_merge']):>7}"
        )
    lines.append("")
    lines.append("Acc = judge accuracy; CWR = confident-wrong rate (wrong & not abstained); SRR-KU = stale-read rate on "
                 "knowledge-update (wrong & returned the superseded value); SRR-KU* = same, restricted to rows whose "
                 "update evidence was in flight; OOCR = out-of-order corruption rate over updated entity keys; "
                 "tokens = mean reader prompt tokens (o200k); readms = median read-path ms excluding reader decode (incl. "
                 "query-time LLM for lazy); #sup = mean committed facts masked per query; MissM = KU missed-merge rate "
                 "(update entity split across two different keys).")
    return "\n".join(lines)
