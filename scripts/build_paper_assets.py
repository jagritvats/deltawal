"""Build every paper table, figure and the numbers manifest from run artifacts. No hand-typed numbers.

Two deterministic stages:

  collect  runs/ (gitignored, large, contains dataset text)  ->  paper/data/snapshot.json.gz (committed)
           slim per-row verdicts (no question/answer/memory text; KU-past is stored as a boolean computed
           with the same regex as the reports), latency benchmark, judge agreement, forensics qids,
           Mem0 history counts, MemConflict audit verdicts, frozen config values, sha256 of every source.
  build    snapshot  ->  paper/tables/*.tex (booktabs), paper/figures/*.pdf (+ preview/*.png, 200 dpi),
           paper/numbers_manifest.json

  uv run python scripts/build_paper_assets.py                  # collect (if runs/ exists) + build
  uv run python scripts/build_paper_assets.py --from-snapshot  # build from the committed snapshot only

The build fails on any matplotlib warning, any overlapping/out-of-figure text, or any box label that does
not fit its box. Bootstrap CIs: 2000 resamples of questions, seed 42 (identical to scripts/report_*.py);
pooled multi-cell / multi-reader cells resample questions as clusters.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PAPER = ROOT / "paper"
SNAP = PAPER / "data" / "snapshot.json.gz"

PAST = re.compile(r"\b(initial|initially|originally|first|before|previous|previously|used to|at first|earlier|prior|"
                  r"than i did|than before|switch|switched)\b", re.I)          # = scripts/report_controls.py

READERS3 = ["gemma4", "gpt-6-luna", "glm-5.3"]
RUN_GROUPS = {
    "heldout": {"gemma4": "m2ho_gemma", "glm-5.3": "m2ho_glm", "gpt-6-luna": "m2ho_luna6"},
    "controls": {"gemma4": "m2ctl_gemma", "glm-5.3": "m2ctl_glm", "gpt-6-luna": "m2ctl_luna6"},
    "unified": {"gemma4": "m2uni_gemma", "glm-5.3": "m2uni_glm", "gpt-6-luna": "m2uni_luna6"},
    "regression60": {"gemma4": "m3r60_gemma", "gpt-6-luna": "m3r60_luna6", "glm-5.3": "m3r60_glm"},
    "mem0": {"gemma4": "m3mem0_gemma", "gpt-6-luna": "m3mem0_luna6", "glm-5.3": "m3mem0_glm"},
    # history hook on all 60 KU questions (its 13 KU-past rows are identical to the earlier m4mem0hist_* runs)
    "mem0hist": {"gemma4": "m6hook60_gemma", "gpt-6-luna": "m6hook60_luna6", "glm-5.3": "m6hook60_glm"},
    "memconflict": {"gemma4": "m3mc_gemma", "gpt-6-luna": "m3mc_luna6", "glm-5.3-flash": "m3mc_glmflash"},
    # read-path ablations of DeltaWAL-Versioned at lag 1 (src/epochmem/systems/ablations.py)
    "ablation": {"gemma4": "m6abl_gemma", "gpt-6-luna": "m6abl_luna6", "glm-5.3": "m6abl_glm"},
    # DeltaWAL-Versioned under reordering (p_ooo = 0.3, lags 1 and 3)
    "reorder": {"gemma4": "m6ooo_gemma", "gpt-6-luna": "m6ooo_luna6", "glm-5.3": "m6ooo_glm"},
    # gemma4 temporal probes (uncached re-run, padded lag 0, sent-date tail, no precedence, tail first)
    "temporal_probe": {"gemma4": "m6temp_gemma"},
    # Table 1 B3 exactly (async LWW index, 1,050-token budget, no tags/precedence) with session dates on raw turns
    "b3dated": {"gemma4": "m6b3d_gemma", "gpt-6-luna": "m6b3d_luna6", "glm-5.3": "m6b3d_glm"},
}
ROW_FIELDS = ["qid", "qtype", "is_abstention", "system", "lag", "p_ooo", "correct", "abstained", "uses_superseded",
              "prompt_tokens", "oocr_updated", "oocr_corrupted", "missed_merge", "cas_rejections",
              "mem0_history_counts", "reader_model", "judge_model"]
SOURCES = {
    "latency": "bench_latency_m2v2_no_reader.json",
    "judge_agreement": "judge_agreement_heldout_full.json",
    "forensics_json": "forensics_heldout.json",
    "forensics_txt": "forensics_heldout.txt",
    "sensitivity_rows": "sensitivity_shadow_heldout_rows.jsonl",
    "mem0_ku_past": "m3mem0_luna6/mem0_ku_past.jsonl",
    "memconflict_audit": "memconflict_audit30.txt",
    "latency_rerun_nondurable": "bench_latency_nondurable_rerun.json",
    "latency_durable": "bench_latency_durable.json",
    "mem0_ku_past_resolved": "mem0_ku_past_resolved.json",
}


# ============================================================================================ collect
def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def collect() -> dict:
    import yaml
    shas, rows = {}, {}
    for group, runs in RUN_GROUPS.items():
        rows[group] = {}
        for reader, run in runs.items():
            p = RUNS / run / "results_judged.jsonl"
            shas[f"runs/{run}/results_judged.jsonl"] = _sha(p)
            out = []
            for r in _jsonl(p):
                s = {k: r.get(k) for k in ROW_FIELDS}
                s["ku_past"] = bool(PAST.search(r["question"]))
                out.append(s)
            rows[group][reader] = sorted(out, key=lambda s: (s["qid"], s["system"], s["lag"], s["p_ooo"]))
    for k, f in SOURCES.items():
        shas[f"runs/{f}"] = _sha(RUNS / f)
    sens = [{**{k: r[k] for k in ("threshold", "qid", "qtype", "is_abstention", "lag", "p_ooo", "correct",
                                   "uses_superseded", "rerun")}, "ku_past": bool(PAST.search(r["question"]))}
            for r in _jsonl(RUNS / SOURCES["sensitivity_rows"])]
    ftxt = (RUNS / SOURCES["forensics_txt"]).read_text(encoding="utf-8", errors="replace").splitlines()
    destructive = {reader: sorted(l.split()[1] for l in ftxt if l.strip().startswith(run) and "(a)" in l)
                   for reader, run in RUN_GROUPS["heldout"].items() if reader != "gemma4"}
    kp = []
    for r in _jsonl(RUNS / SOURCES["mem0_ku_past"]):
        kp.append({"qid": r["qid"], "totals_A": r["totals_A"], "totals_B": r["totals_B"],
                   **{k: v for k, v in r.items() if isinstance(v, bool) or v is None and k.startswith("gold_")}})
    audit = []
    pat = re.compile(r"^\s*(\d+)\s+(mc-\S+)\s+(dynamic|static|conditional)\s+(\S+)")
    for l in (RUNS / SOURCES["memconflict_audit"]).read_text(encoding="utf-8").splitlines():
        m = pat.match(l)
        if m:
            audit.append({"qid": m.group(2), "type": m.group(3), "verdict": m.group(4)})
    cfg = yaml.safe_load((ROOT / "configs/m2_heldout.yaml").read_text(encoding="utf-8"))
    mcfg = yaml.safe_load((ROOT / "configs/m3_mem0.yaml").read_text(encoding="utf-8"))
    shas["configs/m2_heldout.yaml"] = _sha(ROOT / "configs/m2_heldout.yaml")
    shas["configs/m3_mem0.yaml"] = _sha(ROOT / "configs/m3_mem0.yaml")
    a90 = RUNS / "memconflict_audit90.jsonl"          # full audit of the evaluated split (verdicts only; no text)
    audit90 = []
    if a90.exists():
        shas["runs/memconflict_audit90.jsonl"] = _sha(a90)
        audit90 = sorted(({"qid": r["qid"], "type": r["type"], "verdict": r["verdict"]} for r in _jsonl(a90)),
                         key=lambda r: r["qid"])
    a90r2 = RUNS / "memconflict_audit90_r2.jsonl"     # independent second rater (different model), same files
    audit90_r2 = []
    if a90r2.exists():
        shas["runs/memconflict_audit90_r2.jsonl"] = _sha(a90r2)
        audit90_r2 = sorted(({"qid": r["qid"], "type": r["type"], "verdict": r["verdict"]} for r in _jsonl(a90r2)),
                            key=lambda r: r["qid"])
    # extra blind raters from other model families (R3 Codex/GPT, R4 agy/Gemini); same packets and rubric
    audit_extra = {}
    for tag, f in (("r3", "memconflict_audit90_r3_codex.jsonl"), ("r4", "memconflict_audit90_r4_agy.jsonl")):
        if (RUNS / f).exists():
            shas[f"runs/{f}"] = _sha(RUNS / f)
            audit_extra[tag] = sorted(({"qid": r["qid"], "type": r["type"], "verdict": r["verdict"]}
                                       for r in _jsonl(RUNS / f)), key=lambda r: r["qid"])
    # second-family judge (claude-haiku-5-5) on every judged row; verdicts only
    rj = RUNS / "rejudge_haiku" / "rows.jsonl"
    rejudge = []
    if rj.exists():
        shas["runs/rejudge_haiku/rows.jsonl"] = _sha(rj)
        rejudge = sorted(([r["group"], r["reader"], r["qid"], r["system"], r["lag"], r["p_ooo"],
                           int(bool(r["haiku"]["correct"])), int(bool(r["haiku"]["abstained"])),
                           int(bool(r["haiku"]["uses_superseded"]))] for r in _jsonl(rj)))
    # Codex (GPT family) labels: blind static-answer labels and the 50-row judge check
    sl = RUNS / "memconflict_static_labels_codex.jsonl"
    static_labels = []
    if sl.exists():
        shas[f"runs/{sl.name}"] = _sha(sl)
        static_labels = _jsonl(sl)
    jc = RUNS / "judge_check50_codex.csv"
    judge50 = []
    if jc.exists():
        import csv
        shas[f"runs/{jc.name}"] = _sha(jc)
        ref = {r["audit_id"]: r for r in csv.DictReader((RUNS / "human_audit_50.csv").open(encoding="utf-8"))}
        judge50 = [{"audit_id": r["audit_id"], "qid": r["qid"], "stratum": r["stratum"],
                    **{f"codex_{k}": r[f"codex_{k}"] == "True" for k in ("correct", "abstained", "uses_superseded")},
                    **{f"judge_{k}": ref[r["audit_id"]][f"judge_{k}"] == "True"
                       for k in ("correct", "abstained", "uses_superseded")}}
                   for r in csv.DictReader(jc.open(encoding="utf-8"))]
        static_labels = [{k: r[k] for k in ("qid", "run", "system", "lag", "label")} for r in static_labels]
    return {
        "schema": 1,
        "rows": rows,
        "memconflict_audit90_extra": audit_extra,
        "rejudge_haiku": rejudge,
        "memconflict_static_labels_codex": static_labels,
        "judge_check50_codex": judge50,
        "latency_durable": {"nondurable_rerun": json.loads((RUNS / SOURCES["latency_rerun_nondurable"]).read_text(encoding="utf-8")),
                            "durable": json.loads((RUNS / SOURCES["latency_durable"]).read_text(encoding="utf-8"))},
        "mem0_ku_past_resolved": json.loads((RUNS / SOURCES["mem0_ku_past_resolved"]).read_text(encoding="utf-8"))["resolved"],
        "sensitivity": sens,
        "latency": json.loads((RUNS / SOURCES["latency"]).read_text(encoding="utf-8")),
        "extractor_latency": extractor_latency(),
        "judge_agreement": json.loads((RUNS / SOURCES["judge_agreement"]).read_text(encoding="utf-8")),
        "forensics": {"destructive_overwrite_qids": destructive,
                      "summary": json.loads((RUNS / SOURCES["forensics_json"]).read_text(encoding="utf-8"))},
        "mem0_ku_past": kp,
        "memconflict_audit": audit,
        "memconflict_audit90": audit90,
        "memconflict_audit90_r2": audit90_r2,
        "config": {"wal_sim_threshold": cfg["systems"]["wal_sim_threshold"],
                   "shadow_v2_threshold": cfg["systems"]["shadow_v2_threshold"],
                   "wal_top_m": cfg["systems"]["b4_delta_top_m"], "top_k_facts": cfg["memory"]["top_k_facts"],
                   "assistant_trunc_chars_extract": cfg["memory"]["assistant_trunc_chars_extract"],
                   "b3_matched_token_budget": cfg["systems"]["b3_matched_token_budget"],
                   "embedder": cfg["models"]["embedder"], "extractor": cfg["models"]["extractor"],
                   "mem0_version": mcfg["mem0"]["version"], "mem0_llm": mcfg["mem0"]["llm"], "seed": cfg["seed"]},
        "source_sha256": dict(sorted(shas.items())),
    }


def extractor_latency(model: str = "glm-5.3-flash") -> dict:
    """Wall time of the live cloud extraction calls, as logged in the model cache (`_wall_s`). Extraction calls
    are separated from short reader calls with the same model by output length (>= 100 completion tokens)."""
    import sqlite3
    p = ROOT / ".cache" / "llm_cache.sqlite"
    if not p.exists():
        return {}
    walls = []
    with sqlite3.connect(str(p)) as c:
        for (val,) in c.execute("select value from cache where kind='chat' and model=?", (model,)):
            v = json.loads(val)
            if v.get("_wall_s") is not None and ((v.get("usage") or {}).get("completion_tokens") or 0) >= 100:
                walls.append(float(v["_wall_s"]))
    walls.sort()
    if not walls:
        return {}
    med = walls[len(walls) // 2] if len(walls) % 2 else (walls[len(walls) // 2 - 1] + walls[len(walls) // 2]) / 2
    return {"model": model, "n": len(walls), "median_s": round(med, 3),
            "p95_s": round(walls[min(len(walls) - 1, int(0.95 * len(walls)))], 3),
            "note": "live calls only; measured during concurrent batch extraction (up to 24 parallel requests)"}


def write_snapshot(snap: dict) -> None:
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(snap, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as gz:
        gz.write(raw)
    SNAP.write_bytes(buf.getvalue())


def read_snapshot() -> dict:
    with gzip.open(SNAP, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


# ============================================================================================ stats
def boot_ci(vals, iters=2000, seed=42):
    """Identical to scripts/report_heldout.boot_ci (so tables match the text reports)."""
    rng = random.Random(seed)
    n = len(vals)
    means = sorted(sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return sum(vals) / n, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def cluster_boot(rows, f, iters=2000, seed=42):
    """Questions are the resampling unit (several cells/readers per question)."""
    by = defaultdict(list)
    for r in rows:
        by[r["qid"]].append(float(f(r)))
    qs = sorted(by)
    rng = random.Random(seed)
    tot = sum(sum(v) for v in by.values()) / sum(len(v) for v in by.values())
    means = []
    for _ in range(iters):
        s = c = 0.0
        for _ in qs:
            v = by[qs[rng.randrange(len(qs))]]
            s += sum(v)
            c += len(v)
        means.append(s / c)
    means.sort()
    return tot, means[int(0.025 * iters)], means[int(0.975 * iters) - 1]


def stat(rows, f=lambda r: r["correct"], cluster=False):
    if not rows:
        return None
    m, lo, hi = cluster_boot(rows, f) if cluster else boot_ci([float(bool(f(r))) for r in rows])
    # 4 decimals: tables print 1 decimal, and rounding to 2 first would double-round (6/13 = 46.1538 -> 46.15 -> "46.1")
    return {"mean": round(100 * m, 4), "lo": round(100 * lo, 4), "hi": round(100 * hi, 4), "n": len(rows),
            "n_questions": len({r["qid"] for r in rows})}


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def is_ku(r):
    return r["qtype"] == "knowledge-update" and not r["is_abstention"]


def is_tr(r):
    return r["qtype"] == "temporal-reasoning" and not r["is_abstention"]


def stale(r):
    return (not r["correct"]) and bool(r["uses_superseded"])


def cell(rows, system, lag, p=0.0):
    return [r for r in rows if r["system"] == system and r["lag"] == lag and (r["p_ooo"] or 0.0) == p]


def pct(k, n):
    return round(100 * k / n, 2) if n else None


# ---------------------------------------------------------------- paired tests
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a_rows, b_rows, f=lambda r: r["correct"]) -> dict:
    """Paired comparison of two cells on the same questions (joined on qid): A-only / B-only wins, exact McNemar."""
    A, B = {r["qid"]: bool(f(r)) for r in a_rows}, {r["qid"]: bool(f(r)) for r in b_rows}
    q = sorted(set(A) & set(B))
    b = sum(A[k] and not B[k] for k in q)
    c = sum(B[k] and not A[k] for k in q)
    return {"n": len(q), "a_pct": pct(sum(A[k] for k in q), len(q)), "b_pct": pct(sum(B[k] for k in q), len(q)),
            "diff_pp": round(100 * (b - c) / len(q), 2) if q else None, "a_only": b, "b_only": c,
            "p_mcnemar": round(mcnemar_exact(b, c), 4)}


def paired_pooled(pairs: list[tuple[list, list]], f=lambda r: r["correct"], iters=10000, seed=42) -> dict:
    """Pooled over readers: per-reader cells are paired on qid; the p-value is a sign-flip permutation test with
    the question as the unit (all readers' differences on a question flip together)."""
    dq = defaultdict(int)
    b = c = n = 0
    sa = sb = 0
    for a_rows, b_rows in pairs:
        A, B = {r["qid"]: bool(f(r)) for r in a_rows}, {r["qid"]: bool(f(r)) for r in b_rows}
        for k in sorted(set(A) & set(B)):           # sorted: set order varies with hash seeding
            dq[k] += int(A[k]) - int(B[k])
            b += A[k] and not B[k]
            c += B[k] and not A[k]
            sa += A[k]
            sb += B[k]
            n += 1
    d = [dq[k] for k in sorted(dq)]
    obs = abs(sum(d))
    rng = random.Random(seed)
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs for _ in range(iters)) if obs else iters
    return {"n": n, "a_pct": pct(sa, n), "b_pct": pct(sb, n), "diff_pp": round(100 * (b - c) / n, 2) if n else None,
            "a_only": b, "b_only": c, "p_perm": round(hits / iters, 4)}


# ============================================================================================ LaTeX
def tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
            .replace("#", r"\#"))


def tci(s) -> str:
    if s is None:
        return "--"
    return f"{s['mean']:.1f}\\,{{\\scriptsize[{s['lo']:.0f}, {s['hi']:.0f}]}}"


def tnum(x, nd=1) -> str:
    return "--" if x is None else f"{x:.{nd}f}"


def table_env(label, caption, tabulars: list[str], wide=True) -> str:
    env = "table*" if wide else "table"
    body = "\n\\par\\medskip\n".join(tabulars)
    return (f"% generated by scripts/build_paper_assets.py -- do not edit by hand\n"
            f"\\begin{{{env}}}[t]\n\\centering\n\\small\n\\setlength{{\\tabcolsep}}{{4pt}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"{body}\n\\end{{{env}}}\n")


def tabular(colspec, header, body_rows, title=None, ncols=None) -> str:
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule"]
    if title:
        lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textbf{{{title}}}}} \\\\")
        lines.append("\\midrule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    lines.extend(body_rows)
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def hdr2(top: str, bottom: str) -> str:
    """Two-line column header."""
    return f"\\begin{{tabular}}[b]{{@{{}}c@{{}}}}{top}\\\\{bottom}\\end{{tabular}}"


def group_row(text, ncols) -> str:
    return f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{text}}}}} \\\\"


# ============================================================================================ build: tables
SYSNAME = {"B1_sync": "B1 sync (mutable)", "B2_async": "B2 async", "B3_recency_matched_user": "B3 recency (matched)",
           "deltawal_shadow_v2": "Shadow v2", "deltawal_lazy": "Lazy", "deltawal_versioned": "DeltaWAL-Versioned",
           "deltawal_prompt": "DeltaWAL (mutable)", "B1_sync_versioned": "B1 sync (versioned)",
           "Mem0_sync": "Mem0 sync", "Mem0_async": "Mem0 async", "Mem0_deltawal": "Mem0 + DeltaWAL",
           "Mem0_sync_history": "Mem0 sync + history hook"}
READER_DESC = {"gemma4": "gemma4 (12B, local)", "glm-5.3": "glm-5.3 (frontier, cloud)",
               "gpt-6-luna": "gpt-6-luna (cloud)", "glm-5.3-flash": "glm-5.3-flash (cloud)"}


class Builder:
    def __init__(self, snap: dict):
        self.s = snap
        self.R = snap["rows"]
        self.M: dict = {}

    def rec(self, key: str, value):
        assert key not in self.M, key
        self.M[key] = value
        return value

    # ---------------------------------------------------------------- Table 1
    def tab1(self) -> str:
        body, ncols = [], 8
        for reader in ["gemma4", "glm-5.3", "gpt-6-luna"]:
            H, U = self.R["heldout"][reader], self.R["unified"][reader]
            spec = [("B1_sync", 0, cell(H, "B1_sync", 0)), ("B2_async", 1, cell(H, "B2_async", 1)),
                    ("B3_recency_matched_user", 1, cell(H, "B3_recency_matched_user", 1)),
                    ("deltawal_shadow_v2", 1, cell(H, "deltawal_shadow_v2", 1)),
                    ("deltawal_lazy", 1, cell(H, "deltawal_lazy", 1))]
            for lag in (0, 1, 3, 5):
                rs = cell(U, "deltawal_versioned", lag)
                if rs:
                    spec.append(("deltawal_versioned", lag, rs))
            body.append(group_row(f"Reader: {READER_DESC[reader]}", ncols))
            for system, lag, rs in spec:
                assert len(rs) == 138, (reader, system, lag, len(rs))
                ku = [r for r in rs if is_ku(r)]
                st = {"overall": stat(rs), "KU": stat(ku), "temporal": stat([r for r in rs if is_tr(r)]),
                      "abstention": stat([r for r in rs if r["is_abstention"]]),
                      "stale_read_pct_KU": pct(sum(stale(r) for r in ku), len(ku)),
                      "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs), 1)}
                self.rec(f"tab1/{reader}/{system}/lag{lag}", st)
                body.append(f"{tex_escape(SYSNAME[system])} & {lag} & {tci(st['overall'])} & {tci(st['KU'])} & "
                            f"{tci(st['temporal'])} & {tci(st['abstention'])} & {tnum(st['stale_read_pct_KU'])} & "
                            f"{st['mean_prompt_tokens']:.0f} \\\\")
            body.append("\\midrule")
        body.pop()
        n = self.counts_heldout()
        cap = (f"LongMemEval-S held-out set ({n['total']} questions: {n['KU']} knowledge-update, {n['TR']} temporal, "
               f"{n['Abs']} abstention). Accuracy (\\%) with 95\\% bootstrap CIs over questions; stale-read = share of "
               f"KU questions answered with the superseded value; tokens = mean reader prompt tokens. Async systems "
               f"use the deterministic lag barrier with no reordering ($p_\\text{{ooo}}=0$). Every cell is measured "
               f"natively; the judge is qwen3.8-flash throughout.")
        return table_env("tab:main138", cap, [tabular(
            "lrcccccr", ["System", "Lag", "Overall", "KU", "Temporal", "Abstention", "Stale (\\%)", "Tokens"], body)])

    def counts_heldout(self):
        rs = cell(self.R["heldout"]["gemma4"], "B1_sync", 0)
        return {"total": len(rs), "KU": sum(is_ku(r) for r in rs), "TR": sum(is_tr(r) for r in rs),
                "Abs": sum(r["is_abstention"] for r in rs),
                "KU_past": sum(is_ku(r) and r["ku_past"] for r in rs),
                "KU_current": sum(is_ku(r) and not r["ku_past"] for r in rs)}

    # ---------------------------------------------------------------- Table 2
    def tab2(self) -> str:
        cfg = self.s["config"]
        # (a) write side: 2x2 store (in-place vs append-only) x prompt (plain vs epoch-tagged + precedence), lag 0
        da = self.s["forensics"]["destructive_overwrite_qids"]
        rd3 = ["gemma4", "glm-5.3", "gpt-6-luna"]
        spec_a = [("in-place", "plain (B1)", "B1_sync", "heldout"),
                  ("in-place", "epoch tags (DeltaWAL prompt)", "deltawal_prompt", "controls"),
                  ("append-only", "epoch tags (B1 versioned)", "B1_sync_versioned", "controls"),
                  ("append-only", "epoch tags (DeltaWAL-V.)", "deltawal_versioned", "unified")]
        body_a = []
        for store, prompt, system, grp in spec_a:
            per, kp_all, kc_all = {}, [], []
            for reader in rd3:
                rs = cell(self.R[grp][reader], system, 0)
                kp = [r for r in rs if is_ku(r) and r["ku_past"]]
                kc = [r for r in rs if is_ku(r) and not r["ku_past"]]
                q = set(da.get(reader, []))
                st = {"KU_past": stat(kp), "KU_current": stat(kc),
                      "destructive_cases": {"correct": sum(1 for r in rs if r["qid"] in q and r["correct"]), "n": len(q)}
                      if q else None}
                self.rec(f"tab2a/{reader}/{system}", st)
                per[reader] = st
                kp_all += kp
                kc_all += kc
            pooled = {"KU_past": stat(kp_all, cluster=True), "KU_current": stat(kc_all, cluster=True)}
            self.rec(f"tab2a/pooled/{system}", pooled)
            body_a.append(f"{store} & {prompt} & {tci(pooled['KU_past'])} & "
                          + " / ".join(tnum(per[rd]["KU_past"]["mean"]) for rd in rd3)
                          + f" & {tci(pooled['KU_current'])} \\\\")
        tot = {s: sum((self.M[f"tab2a/{rd}/{s}"]["destructive_cases"] or {}).get("correct", 0) for rd in rd3)
               for s in ("B1_sync", "deltawal_versioned")}
        n_destr = sum(len(v) for v in da.values())
        self.rec("tab2a/destructive_cases_combined", {"mutable": {"correct": tot["B1_sync"], "n": n_destr},
                                                      "versioned": {"correct": tot["deltawal_versioned"], "n": n_destr}})
        nh = self.counts_heldout()
        ta = tabular("llccc", ["Store", "Prompt", hdr2("KU-past, pooled", f"($n$={nh['KU_past']} per reader)"),
                               hdr2("KU-past per reader", "(gemma4 / glm / gpt)"),
                               hdr2("KU-current, pooled", f"($n$={nh['KU_current']} per reader)")], body_a,
                     title="(a) Write side at lag 0: store policy and prompt format", ncols=5)
        # (b) read side
        sw, H = self.shadow_sweep(), None
        body_b = []
        for label, d in sw:
            body_b.append(f"{label} & {d['cells_rerun'] if d['cells_rerun'] is not None else '--'} & {tci(d['overall'])} & "
                          f"{tci(d['KU'])} & {tci(d['KU_past'])} & {tci(d['KU_current'])} \\\\")
        tb = tabular("lrcccc", ["Threshold", "Re-run", "Overall", "KU", "KU-past", "KU-current"], body_b,
                     title=f"(b) Read side: Shadow v2 masking threshold (gemma4; frozen {cfg['shadow_v2_threshold']:.2f})",
                     ncols=6)
        # (c) integrity at p_ooo = 0.3
        body_c = []
        H = self.R["heldout"]["gemma4"]
        for lag in (1, 3):
            for policy, system in (("LWW (B2, B3)", "B2_async"), ("CAS-fenced (DeltaWAL)", "deltawal_prompt")):
                rs = cell(H, system, lag, 0.3)
                upd = sum(r["oocr_updated"] or 0 for r in rs)
                cor = sum(r["oocr_corrupted"] or 0 for r in rs)
                cas = sum(r["cas_rejections"] or 0 for r in rs)
                ku = [r for r in rs if is_ku(r)]
                mm = sum(bool(r["missed_merge"]) for r in ku)
                st = {"updated_keys": upd, "corrupted_keys": cor, "oocr_pct": pct(cor, upd), "cas_rejections": cas,
                      "questions_with_missed_merge": mm, "n_questions": len(ku),
                      "missed_merge_pct": pct(mm, len(ku)),
                      "all_questions_flagged": sum(bool(r["missed_merge"]) for r in rs), "all_questions": len(rs)}
                self.rec(f"tab2c/lag{lag}/{system}", st)
                body_c.append(f"{lag} & {policy} & {upd} & {cor} ({st['oocr_pct']:.1f}\\%) & {cas} & "
                              f"{mm}/{len(ku)} ({st['missed_merge_pct']:.1f}\\%) \\\\")
        tc = tabular("rlrrrr", ["Lag", "Store policy", hdr2("Updated", "keys"), hdr2("Corrupted", "keys (OOCR)"),
                                hdr2("CAS", "rejections"), hdr2("KU questions with", "missed merge")], body_c,
                     title="(c) Store integrity under reordering ($p_\\text{ooo}=0.3$, store-level)", ncols=6)
        cap = ("History ablations on the 138 held-out questions. (a) Store policy and prompt format at lag 0: "
               "B1 sync (in-place, plain dated facts), the DeltaWAL prompt over its CAS store (in-place, epoch tags and "
               "precedence instruction), B1 sync over an append-only store with the same kind of epoch-tagged prompt, "
               "and DeltaWAL-Versioned (empty WAL); rows 2 and 3 differ only in the store, rows 1 and 2 only in the "
               "prompt. Pooled CIs resample questions. (b) Shadow v2 masking threshold, pooled over the five "
               "async cells (lag 1/3/5, $p_\\text{ooo}\\in\\{0, 0.3\\}$); re-run = question-cells whose set of masked "
               "facts differs from the frozen threshold's and therefore needed a new reader call (the others keep the "
               "frozen-threshold answer); `off' is the unmasked epoch-merge prompt on the same cells; CIs resample "
               "questions. (c) Out-of-order corruption (an updated key left at an older "
               "epoch) vs.\\ missed merges, counted on the knowledge-update questions: the update's facts share no key "
               "with the earlier evidence, so no keyed policy can supersede the old value.")
        return table_env("tab:history", cap, [ta, tb, tc])

    def shadow_sweep(self):
        if hasattr(self, "_sweep"):
            return self._sweep
        S = self.s["sensitivity"]
        H = self.R["heldout"]["gemma4"]
        out = []
        cells = sorted({(r["lag"], r["p_ooo"]) for r in S})
        for t in sorted({r["threshold"] for r in S}):
            rs = [r for r in S if r["threshold"] == t]
            out.append((f"{t:.2f}", rs, sum(r["rerun"] for r in rs)))
        off = [r for r in H if r["system"] == "deltawal_prompt" and (r["lag"], r["p_ooo"]) in set(cells)]
        out.append(("off (unmasked)", off, None))
        res = []
        for label, rs, rerun in out:
            d = {"cells": len(rs), "cells_rerun": rerun, "overall": stat(rs, cluster=True),
                 "KU": stat([r for r in rs if is_ku(r)], cluster=True),
                 "KU_past": stat([r for r in rs if is_ku(r) and r["ku_past"]], cluster=True),
                 "KU_current": stat([r for r in rs if is_ku(r) and not r["ku_past"]], cluster=True)}
            self.rec(f"tab2b/threshold_{label.split()[0]}", d)
            res.append((label, d))
        self.rec("tab2b/pooled_cells", [list(c) for c in cells])
        self._sweep = res
        return res

    # ---------------------------------------------------------------- Table 3
    def tab3(self) -> str:
        strata = [("All", lambda r: True), ("multi-session", lambda r: r["qtype"] == "multi-session"),
                  ("single-session-user", lambda r: r["qtype"] == "single-session-user"),
                  ("single-session-assistant", lambda r: r["qtype"] == "single-session-assistant"),
                  ("single-session-preference", lambda r: r["qtype"] == "single-session-preference")]
        spec = [("B1_sync", 0), ("deltawal_versioned", 0), ("B2_async", 1), ("B3_recency_matched_user", 1),
                ("deltawal_versioned", 1)]
        body, ncols, ns = [], 8, None
        for reader in READERS3:
            rows = self.R["regression60"][reader]
            body.append(group_row(f"Reader: {READER_DESC[reader]}", ncols))
            for system, lag in spec:
                rs = cell(rows, system, lag)
                if not rs:
                    continue
                assert len(rs) == 60
                ns = ns or {k: sum(1 for r in rs if f(r)) for k, f in strata}
                st = {k: stat([r for r in rs if f(r)]) for k, f in strata}
                st["mean_prompt_tokens"] = round(sum(r["prompt_tokens"] for r in rs) / len(rs), 1)
                self.rec(f"tab3/{reader}/{system}/lag{lag}", st)
                body.append(f"{tex_escape(SYSNAME[system])} & {lag} & " + " & ".join(tci(st[k]) for k, _ in strata) +
                            f" & {st['mean_prompt_tokens']:.0f} \\\\")
            body.append("\\midrule")
        body.pop()
        self.rec("datasets/regression60", ns)
        hdr = ["System", "Lag"] + [hdr2(k.replace("single-session-", "SS-").replace("multi-session", "Multi-session"),
                                        f"($n$={ns[k]})") for k, _ in strata] + ["Tokens"]
        c = self.s["config"]
        cap = (f"Non-update regression: {ns['All']} LongMemEval-S questions disjoint from the held-out set. The "
               f"extractor sees assistant turns truncated to {c['assistant_trunc_chars_extract']} characters, so "
               f"fact-only memories rarely contain assistant-provided details (SS-assistant); DeltaWAL recovers them "
               f"from the raw WAL turn when the session is in flight. SS-preference uses LongMemEval's rubric grading. "
               f"glm-5.3 was run only on the B1 and DeltaWAL-Versioned cells.")
        return table_env("tab:regression60", cap, [tabular("lrcccccr", hdr, body)])

    # ---------------------------------------------------------------- Table 4
    def tab4(self) -> str:
        body, ncols = [], 6
        for reader in READERS3:
            rows, hist = self.R["mem0"][reader], self.R["mem0hist"][reader]
            body.append(group_row(f"Reader: {READER_DESC[reader]}", ncols))
            for system, lag, rs in (("Mem0_sync", 0, cell(rows, "Mem0_sync", 0)),
                                    ("Mem0_async", 1, cell(rows, "Mem0_async", 1)),
                                    ("Mem0_deltawal", 1, cell(rows, "Mem0_deltawal", 1)),
                                    ("Mem0_sync_history", 0, cell(hist, "Mem0_sync_history", 0))):
                if not rs:
                    continue
                past = [r for r in rs if r["ku_past"]]
                cur = [r for r in rs if not r["ku_past"]]
                full = len(rs) == 60
                st = {"KU": stat(rs) if full else None, "KU_past": stat(past), "KU_current": stat(cur) if full else None,
                      "stale_read_pct": pct(sum(stale(r) for r in rs), len(rs)), "n_rows": len(rs)}
                self.rec(f"tab4/{reader}/{system}/lag{lag}", st)
                stale_txt = tnum(st["stale_read_pct"]) if full else f"{tnum(st['stale_read_pct'])}$^\\ast$"
                body.append(f"{tex_escape(SYSNAME[system])} & {lag} & {tci(st['KU'])} & {tci(st['KU_past'])} & "
                            f"{tci(st['KU_current'])} & {stale_txt} \\\\")
            body.append("\\midrule")
        body.pop()
        n_past = sum(r["ku_past"] for r in cell(self.R["mem0"]["gemma4"], "Mem0_sync", 0))
        t1 = tabular("lrcccc", ["System", "Lag", hdr2("KU", "($n$=60)"), hdr2("KU-past", f"($n$={n_past})"),
                                hdr2("KU-current", f"($n$={60 - n_past})"), hdr2("Stale", "(\\%)")], body)
        # history events (store-level, identical for every reader)
        ref = self.R["mem0"]["gpt-6-luna"]
        ev = {}
        for label, system, lag in (("sessions $1..T{-}1$ (async view)", "Mem0_async", 1),
                                   ("sessions $1..T$ (sync view)", "Mem0_sync", 0)):
            tot = defaultdict(int)
            for r in cell(ref, system, lag):
                for k, v in (r["mem0_history_counts"] or {}).items():
                    tot[k] += v
            ev[label] = dict(tot)
        kp = self.s["mem0_ku_past"]
        ktot = defaultdict(int)
        for r in kp:
            for k, v in r["totals_B"].items():
                ktot[k] += v
        ev[f"KU-past subset, sessions $1..T$ ($n$={len(kp)})"] = dict(ktot)
        self.rec("tab4/history_events", {k.replace("$", "").replace("{-}", "-"): v for k, v in ev.items()})
        flags = {k: sum(1 for r in kp if r.get(k)) for k in ("old_value_updated_in_place", "old_value_deleted",
                                                               "gold_old_value_in_live_store",
                                                               "gold_old_value_only_in_history")}
        flags["n"] = len(kp)
        # the 5 undetermined flags, resolved by searching the persisted stores (runs/mem0_ku_past_resolved.json)
        res = self.s.get("mem0_ku_past_resolved") or {}
        fin = {}
        for r in kp:
            fin[r["qid"]] = ("live" if r.get("gold_old_value_in_live_store") else
                             "history_only" if r.get("gold_old_value_only_in_history") else
                             res.get(r["qid"], {}).get("flag", "undetermined"))
        flags["resolved"] = {k: sum(v == k for v in fin.values()) for k in ("live", "history_only", "undetermined")}
        self.rec("tab4/ku_past_flags", flags)
        b2 = [f"{label} & {v.get('ADD', 0)} & {v.get('UPDATE', 0)} & {v.get('DELETE', 0)} \\\\" for label, v in ev.items()]
        t2 = tabular("lrrr", ["Mem0 history events (60 questions)", "ADD", "UPDATE", "DELETE"], b2)
        c = self.s["config"]
        cap = (f"Mem0 OSS {c['mem0_version']} (classic ADD/UPDATE/DELETE write path) on the 60 held-out "
               f"knowledge-update questions. Mem0 sync ingests every session; Mem0 async leaves the final (update) "
               f"session in flight; Mem0 + DeltaWAL adds the epoch-tagged WAL and relevance gate "
               f"($\\tau={c['wal_sim_threshold']}$) on top of the async store. The history hook renders, for every "
               f"retrieved memory, its earlier versions from \\texttt{{Memory.history()}} in epoch order. On the "
               f"{flags['n']} KU-past questions Mem0 updated the older evidence memory in place in "
               f"{flags['old_value_updated_in_place']} and deleted it in {flags['old_value_deleted']}; the old value "
               f"survived only in the history table in {flags['resolved']['history_only']}.")
        return table_env("tab:mem0", cap, [t1, t2])

    # ---------------------------------------------------------------- Table 5
    def tab5(self) -> str:
        strata = [("All", lambda r: True), ("Dynamic", lambda r: r["qtype"] == "memconflict-dynamic"),
                  ("Static", lambda r: r["qtype"] == "memconflict-static"),
                  ("Conditional", lambda r: r["qtype"] == "memconflict-conditional")]
        spec = [("B1_sync", 0), ("deltawal_versioned", 0), ("B2_async", 1), ("B3_recency_matched_user", 1),
                ("deltawal_versioned", 1)]
        body, ncols, ns = [], 7, None
        for reader in ["gemma4", "gpt-6-luna", "glm-5.3-flash"]:
            rows = self.R["memconflict"][reader]
            body.append(group_row(f"Reader: {READER_DESC[reader]}", ncols))
            for system, lag in spec:
                rs = cell(rows, system, lag)
                assert len(rs) == 90, (reader, system, lag, len(rs))
                ns = ns or {k: sum(1 for r in rs if f(r)) for k, f in strata}
                st = {k: stat([r for r in rs if f(r)]) for k, f in strata}
                stat_rows = [r for r in rs if r["qtype"] == "memconflict-static"]
                st["static_false_belief"] = stat(stat_rows, f=stale)
                st["dynamic_superseded"] = stat([r for r in rs if r["qtype"] == "memconflict-dynamic"], f=stale)
                self.rec(f"tab5/{reader}/{system}/lag{lag}", st)
                body.append(f"{tex_escape(SYSNAME[system])} & {lag} & " + " & ".join(tci(st[k]) for k, _ in strata) +
                            f" & {tci(st['static_false_belief'])} \\\\")
            body.append("\\midrule")
        body.pop()
        self.rec("datasets/memconflict", ns)
        a = self.s["memconflict_audit"]
        clean = sum(x["verdict"] == "clean" for x in a)
        lo, hi = wilson(clean, len(a))
        self.rec("memconflict_audit", {"n": len(a), "clean": clean, "clean_pct": pct(clean, len(a)),
                                       "wilson95": [round(100 * lo, 1), round(100 * hi, 1)], "gate_pct": 85,
                                       "passed": 100 * clean / len(a) >= 85})
        hdr = ["System", "Lag"] + [hdr2(k, f"($n$={ns[k]})") for k, _ in strata] + [hdr2("Static: believed", "false statement")]
        cap = (f"MemConflict ({ns['All']} questions, {ns['Dynamic']}/{ns['Static']}/{ns['Conditional']} "
               f"dynamic/static/conditional). A pre-evaluation audit of {len(a)} questions sampled from the whole "
               f"pool (LLM reading agent) found {clean}/{len(a)} clean; the later full audit of these evaluated "
               f"questions found fewer and misses the $\\geq$85\\% gate (Section~\\ref{{sec:memconflict}}, "
               f"Table~\\ref{{tab:memconflict_clean}}). "
               f"`Static: believed false' = static questions answered with the later, false statement. In the async "
               f"cells the false statement sits in the in-flight session, which B2/B3 never see.")
        return table_env("tab:memconflict", cap, [tabular("lrccccc", hdr, body)])

    # ---------------------------------------------------------------- Table 6
    def tab7(self) -> str:
        """Table 5 re-scored on the evaluated MemConflict questions that the full audit found clean."""
        a90 = self.s.get("memconflict_audit90") or []
        if not a90:
            return "% no full MemConflict audit in the snapshot\n"
        verdict = {r["qid"]: r["verdict"] for r in a90}
        evaluated = {r["qid"] for r in cell(self.R["memconflict"]["gemma4"], "B1_sync", 0)}
        assert set(verdict) == evaluated, "audit90 must cover exactly the evaluated split"
        types = ("dynamic", "static", "conditional")
        cnt = {t: {v: sum(1 for r in a90 if r["type"] == t and r["verdict"] == v)
                   for v in ("clean", "ambiguous", "label_error")} for t in types}
        clean_n = sum(c["clean"] for c in cnt.values())
        lo, hi = wilson(clean_n, len(a90))
        self.rec("memconflict_audit90", {"n": len(a90), "clean": clean_n, "clean_pct": pct(clean_n, len(a90)),
                                         "not_clean": len(a90) - clean_n,
                                         "wilson95": [round(100 * lo, 1), round(100 * hi, 1)], "by_type": cnt,
                                         "gate_pct": 85, "passed_on_lower_bound": 100 * lo >= 85})
        r2v = {r["qid"]: r["verdict"] for r in (self.s.get("memconflict_audit90_r2") or [])}
        keep = {q for q, v in verdict.items() if v == "clean" and (not r2v or r2v.get(q) == "clean")}
        strata = [("All", lambda r: True), ("Dynamic", lambda r: r["qtype"] == "memconflict-dynamic"),
                  ("Static", lambda r: r["qtype"] == "memconflict-static"),
                  ("Conditional", lambda r: r["qtype"] == "memconflict-conditional")]
        spec = [("B1_sync", 0), ("deltawal_versioned", 0), ("B2_async", 1), ("B3_recency_matched_user", 1),
                ("deltawal_versioned", 1)]
        body, ncols, ns = [], 7, None
        for reader in ["gemma4", "gpt-6-luna", "glm-5.3-flash"]:
            rows = [r for r in self.R["memconflict"][reader] if r["qid"] in keep]
            body.append(group_row(f"Reader: {READER_DESC[reader]}", ncols))
            for system, lag in spec:
                rs = cell(rows, system, lag)
                ns = ns or {k: sum(1 for r in rs if f(r)) for k, f in strata}
                st = {k: stat([r for r in rs if f(r)]) for k, f in strata}
                st["static_false_belief"] = stat([r for r in rs if r["qtype"] == "memconflict-static"], f=stale)
                self.rec(f"tab7/{reader}/{system}/lag{lag}", st)
                body.append(f"{tex_escape(SYSNAME[system])} & {lag} & " + " & ".join(tci(st[k]) for k, _ in strata) +
                            f" & {tci(st['static_false_belief'])} \\\\")
            body.append("\\midrule")
        body.pop()
        pooled = {}
        for system, lag in spec:
            rows = [r for rd in ["gemma4", "gpt-6-luna", "glm-5.3-flash"]
                    for r in cell(self.R["memconflict"][rd], system, lag) if r["qid"] in keep]
            pooled[f"{system}/lag{lag}"] = {
                "dynamic": stat([r for r in rows if r["qtype"] == "memconflict-dynamic"], cluster=True),
                "static_false_belief": stat([r for r in rows if r["qtype"] == "memconflict-static"], f=stale,
                                            cluster=True)}
        self.rec("tab7/pooled", pooled)
        self.rec("datasets/memconflict_clean", ns)
        r2 = {r["qid"]: r["verdict"] for r in (self.s.get("memconflict_audit90_r2") or [])}
        if r2:
            assert set(r2) == set(verdict)
            q = sorted(verdict)

            def kappa(x, y):
                po = sum(a == b for a, b in zip(x, y)) / len(x)
                pe = sum((x.count(c) / len(x)) * (y.count(c) / len(y)) for c in set(x) | set(y))
                return po, (po - pe) / (1 - pe)
            bx = ["clean" if verdict[k] == "clean" else "flagged" for k in q]
            by = ["clean" if r2[k] == "clean" else "flagged" for k in q]
            po2, k2 = kappa(bx, by)
            po3, k3 = kappa([verdict[k] for k in q], [r2[k] for k in q])
            both = {k for k in q if verdict[k] == "clean" and r2[k] == "clean"}
            either = {k for k in q if verdict[k] == "clean" or r2[k] == "clean"}
            rob = {}
            for label, keepset in (("both_clean", both), ("either_clean", either),
                                   ("r2_clean", {k for k in q if r2[k] == "clean"})):
                for system, lag in (("B1_sync", 0), ("deltawal_versioned", 0), ("B2_async", 1)):
                    rows = [r for rd in ["gemma4", "gpt-6-luna", "glm-5.3-flash"]
                            for r in cell(self.R["memconflict"][rd], system, lag) if r["qid"] in keepset]
                    rob[f"{label}/{system}/lag{lag}"] = {
                        "dynamic": stat([r for r in rows if r["qtype"] == "memconflict-dynamic"], cluster=True),
                        "static_false_belief": stat([r for r in rows if r["qtype"] == "memconflict-static"],
                                                    f=stale, cluster=True)}
            self.rec("memconflict_audit90_r2", {
                "clean": sum(v == "clean" for v in r2.values()), "n": len(r2),
                "clean_pct": pct(sum(v == "clean" for v in r2.values()), len(r2)),
                "agreement_binary_pct": round(100 * po2, 1), "kappa_binary": round(k2, 3),
                "agreement_3way_pct": round(100 * po3, 1), "kappa_3way": round(k3, 3),
                "both_clean": len(both), "either_clean": len(either),
                "disagreements": len(q) - round(po2 * len(q)), "robustness": rob})
        hdr = ["System", "Lag"] + [hdr2(k, f"($n$={ns[k]})") for k, _ in strata] + [hdr2("Static: believed", "false statement")]
        cap = (f"MemConflict re-scored on the {len(keep)} of {len(a90)} evaluated questions that both raters "
               f"found clean (Appendix~\\ref{{app:memconflict}}). Same judged answers as "
               f"Table~\\ref{{tab:memconflict}}; only the question set changes.")
        return table_env("tab:memconflict_clean", cap, [tabular("lrccccc", hdr, body)])

    def tab8(self) -> str:
        """Audit counts on the evaluated split: four blind raters from three model families."""
        a1 = {r["qid"]: r for r in self.s.get("memconflict_audit90") or []}
        a2 = {r["qid"]: r["verdict"] for r in self.s.get("memconflict_audit90_r2") or []}
        ex = self.s.get("memconflict_audit90_extra") or {}
        a3 = {r["qid"]: r["verdict"] for r in ex.get("r3", [])}
        a4 = {r["qid"]: r["verdict"] for r in ex.get("r4", [])}
        if not a1 or not a2 or not a3 or not a4:
            return "% audit data missing from the snapshot\n"
        V = {"R1": {q: r["verdict"] for q, r in a1.items()}, "R2": a2, "R3": a3, "R4": a4}
        types = ("dynamic", "static", "conditional")
        rows, rec = [], {}
        for t in types + ("all",):
            q = [k for k, r in a1.items() if t == "all" or r["type"] == t]
            e = {"n": len(q), **{f"clean_{k.lower()}": sum(V[k][x] == "clean" for x in q) for k in V},
                 "clean_r1r2": sum(V["R1"][x] == "clean" and V["R2"][x] == "clean" for x in q),
                 "clean_all4": sum(all(V[k][x] == "clean" for k in V) for x in q),
                 "clean_majority": sum(sum(V[k][x] == "clean" for k in V) >= 3 for x in q)}
            rec[t] = {"evaluated": e}
            if t == "all":
                rows.append("\\midrule")
            rows.append(f"{t.capitalize()} & {e['n']} & {e['clean_r1']} & {e['clean_r2']} & {e['clean_r3']} & "
                        f"{e['clean_r4']} & {e['clean_r1r2']} & {e['clean_all4']} \\\\")
        rec["static_evaluated_clean_both_pct"] = pct(rec["static"]["evaluated"]["clean_r1r2"], rec["static"]["evaluated"]["n"])
        for t in types:                                    # keep the earlier key names used in the text
            rec[t]["evaluated"]["clean_both"] = rec[t]["evaluated"]["clean_r1r2"]
        q = sorted(a1)
        bin_ = {k: [V[k][x] == "clean" for x in q] for k in V}

        def kap(x, y):
            n = len(x)
            po = sum(a == b for a, b in zip(x, y)) / n
            pa, pb = sum(x) / n, sum(y) / n
            pe = pa * pb + (1 - pa) * (1 - pb)
            return round((po - pe) / (1 - pe), 3)
        pairs = {f"{a}_{b}": kap(bin_[a], bin_[b]) for i, a in enumerate(V) for b in list(V)[i + 1:]}
        # Fleiss' kappa, binary
        N, m = len(q), len(V)
        nij = [[sum(bin_[k][i] for k in V), m - sum(bin_[k][i] for k in V)] for i in range(N)]
        P = [(sum(c * c for c in row) - m) / (m * (m - 1)) for row in nij]
        pj = [sum(row[j] for row in nij) / (N * m) for j in (0, 1)]
        pe = sum(x * x for x in pj)
        rec["kappa_pairs"] = pairs
        rec["kappa_fleiss4"] = round((sum(P) / N - pe) / (1 - pe), 3)
        self.rec("tab8", rec)
        hdr = ["Type", "$n$", "R1", "R2", "R3", "R4", hdr2("R1 and R2", "(main set)"), hdr2("All four", "raters")]
        cap = ("MemConflict audit of the evaluated split: questions rated clean by each blind rater (R1 and R2 on "
               "Claude models, R3 on a GPT model via Codex CLI, R4 on a Gemini model via Antigravity), by R1 and R2 "
               "(the set used in Table~\\ref{tab:memconflict_clean}) and by all four. Fleiss' $\\kappa$ (clean vs.\\ "
               f"not clean) is {rec['kappa_fleiss4']:.3f}.")
        return table_env("tab:memconflict_audit", cap, [tabular("lrrrrrrr", hdr, rows)])

    # ---------------------------------------------------------------- paired tests and second-judge robustness
    def _haiku(self, group, reader, rows):
        """The same rows with the second-family judge's verdicts (claude-haiku-5-5). Each row's own group and reader
        are looked up (cells can mix groups), so `group` is only a fallback."""
        if not hasattr(self, "_hz"):
            self._hz = {(g, rd, q, s, l, float(p or 0.0)): (c, a, u)
                        for g, rd, q, s, l, p, c, a, u in self.s.get("rejudge_haiku") or []}
            self._grp = {id(r): (g, rd) for g, runs in self.R.items() for rd, rs in runs.items() for r in rs}
        out = []
        for r in rows:
            g, rd = self._grp.get(id(r), (group, reader))
            v = self._hz.get((g, rd, r["qid"], r["system"], r["lag"], float(r["p_ooo"] or 0.0)))
            if v is None:
                return None
            out.append({**r, "correct": bool(v[0]), "abstained": bool(v[1]), "uses_superseded": bool(v[2])})
        return out

    def _clean70(self):
        a1 = {r["qid"]: r["verdict"] for r in self.s.get("memconflict_audit90") or []}
        a2 = {r["qid"]: r["verdict"] for r in self.s.get("memconflict_audit90_r2") or []}
        return {q for q in a1 if a1[q] == "clean" and a2.get(q) == "clean"}

    def tests(self) -> None:
        R, rd3 = self.R, ["gemma4", "glm-5.3", "gpt-6-luna"]
        U = lambda rd, s, l, p=0.0: cell(R["unified"][rd], s, l, p)            # noqa: E731
        H = lambda rd, s, l, p=0.0: cell(R["heldout"][rd], s, l, p)            # noqa: E731
        C = lambda rd, s, l: cell(R["controls"][rd], s, l)                     # noqa: E731
        ku = lambda rs: [r for r in rs if is_ku(r)]                            # noqa: E731
        kp = lambda rs: [r for r in rs if is_ku(r) and r["ku_past"]]           # noqa: E731
        kc = lambda rs: [r for r in rs if is_ku(r) and not r["ku_past"]]       # noqa: E731
        tr = lambda rs: [r for r in rs if is_tr(r)]                            # noqa: E731

        def both(name, pairs, f=lambda r: r["correct"], groups=None):
            """Per-reader exact McNemar + pooled permutation; optionally the same under the haiku judge."""
            out = {rd: paired(a, b, f) for rd, (a, b) in pairs.items()}
            out["pooled"] = paired_pooled(list(pairs.values()), f)
            groups = groups or ("?", "?")                   # rows carry their own group (see _haiku)
            if groups:
                hz = {}
                for rd, (a, b) in pairs.items():
                    ha, hb = self._haiku(groups[0], rd, a), self._haiku(groups[1], rd, b)
                    if ha is None or hb is None:
                        hz = None
                        break
                    hz[rd] = (ha, hb)
                if hz:
                    out["haiku"] = {rd: paired(a, b, f) for rd, (a, b) in hz.items()}
                    out["haiku"]["pooled"] = paired_pooled(list(hz.values()), f)
            self.rec(f"paired/{name}", out)

        # Section 3: staleness and lag
        both("ku_dwv1_vs_b2", {rd: (ku(U(rd, "deltawal_versioned", 1)), ku(H(rd, "B2_async", 1))) for rd in rd3},
             groups=("unified", "heldout"))
        both("ku_dwv1_vs_b1", {rd: (ku(U(rd, "deltawal_versioned", 1)), ku(H(rd, "B1_sync", 0))) for rd in rd3},
             groups=("unified", "heldout"))
        both("ku_dwv1_vs_b3", {rd: (ku(U(rd, "deltawal_versioned", 1)), ku(H(rd, "B3_recency_matched_user", 1)))
                               for rd in rd3})
        both("tr_gemma_dwv1_vs_dwv0", {"gemma4": (tr(U("gemma4", "deltawal_versioned", 1)),
                                                  tr(U("gemma4", "deltawal_versioned", 0)))})
        # Section 4: store vs prompt (2x2 at lag 0), read-side erasure
        # store effect with the prompt held fixed (both epoch-tagged + precedence): append-only vs in-place (CAS)
        both("kupast_store_effect", {rd: (kp(C(rd, "B1_sync_versioned", 0)), kp(C(rd, "deltawal_prompt", 0))) for rd in rd3})
        # the same store effect on KU-current (does keeping versions cost current-value questions?)
        both("kucur_store_effect", {rd: (kc(C(rd, "B1_sync_versioned", 0)), kc(C(rd, "deltawal_prompt", 0))) for rd in rd3})
        both("kupast_prompt_effect", {rd: (kp(C(rd, "deltawal_prompt", 0)), kp(H(rd, "B1_sync", 0))) for rd in rd3})
        both("kupast_dwv0_vs_b1", {rd: (kp(U(rd, "deltawal_versioned", 0)), kp(H(rd, "B1_sync", 0))) for rd in rd3},
             groups=("unified", "heldout"))
        both("kupast_dwv1_vs_shadow", {rd: (kp(U(rd, "deltawal_versioned", 1)), kp(H(rd, "deltawal_shadow_v2", 1)))
                                       for rd in rd3}, groups=("unified", "heldout"))
        both("kupast_dwv1_vs_lazy", {rd: (kp(U(rd, "deltawal_versioned", 1)), kp(H(rd, "deltawal_lazy", 1)))
                                     for rd in rd3}, groups=("unified", "heldout"))
        both("kucur_dwv1_vs_shadow", {rd: (kc(U(rd, "deltawal_versioned", 1)), kc(H(rd, "deltawal_shadow_v2", 1)))
                                      for rd in rd3})
        # Section 5: Mem0 hook (all 60 KU), MemConflict (clean 70), regression set
        M, MH = R["mem0"], R["mem0hist"]
        for nm, sel in (("ku", lambda rs: rs), ("kupast", lambda rs: [r for r in rs if r["ku_past"]]),
                        ("kucur", lambda rs: [r for r in rs if not r["ku_past"]])):
            both(f"mem0_hook_vs_sync_{nm}", {rd: (sel(cell(MH[rd], "Mem0_sync_history", 0)),
                                                  sel(cell(M[rd], "Mem0_sync", 0))) for rd in rd3},
                 groups=("mem0hist", "mem0") if nm == "kupast" else None)
        keep, mc3 = self._clean70(), ["gemma4", "gpt-6-luna", "glm-5.3-flash"]
        mcs = lambda rd, s, l, t: [r for r in cell(R["memconflict"][rd], s, l)          # noqa: E731
                                   if r["qid"] in keep and r["qtype"] == f"memconflict-{t}"]
        both("mc_dynamic_dwv0_vs_b1", {rd: (mcs(rd, "deltawal_versioned", 0, "dynamic"), mcs(rd, "B1_sync", 0, "dynamic"))
                                       for rd in mc3}, groups=("memconflict", "memconflict"))
        both("mc_static_falsebelief_dwv0_vs_b1", {rd: (mcs(rd, "deltawal_versioned", 0, "static"),
                                                       mcs(rd, "B1_sync", 0, "static")) for rd in mc3},
             f=stale, groups=("memconflict", "memconflict"))
        # the same static comparison under Codex's blind labels (false_value = believed the later, false statement)
        sl = self.s.get("memconflict_static_labels_codex") or []
        if sl:
            run2rd = {v: k for k, v in RUN_GROUPS["memconflict"].items()}
            ex = self.s.get("memconflict_audit90_extra") or {}
            v3 = {r["qid"]: r["verdict"] for r in ex.get("r3", [])}
            v4 = {r["qid"]: r["verdict"] for r in ex.get("r4", [])}
            all4 = {q for q in keep if v3.get(q) == "clean" and v4.get(q) == "clean"}
            lab = defaultdict(list)
            for r in sl:
                lab[(run2rd[r["run"]], r["system"])].append({"qid": r["qid"], "fb": r["label"] == "false_value",
                                                            "fb_or_both": r["label"] in ("false_value", "both")})
            for nm, qs in (("clean70", keep), ("all4clean", all4)):
                for fld in ("fb", "fb_or_both"):
                    pr = {rd: ([x for x in lab[(rd, "deltawal_versioned")] if x["qid"] in qs],
                               [x for x in lab[(rd, "B1_sync")] if x["qid"] in qs]) for rd in mc3}
                    self.rec(f"paired/mc_static_codex_{fld}_{nm}", {"pooled": paired_pooled(list(pr.values()), f=lambda x: x[fld]),
                                                                    "n_questions": len(qs & {x["qid"] for x in sl})})
        both("mc_conditional_dwv0_vs_b1", {rd: (mcs(rd, "deltawal_versioned", 0, "conditional"),
                                                mcs(rd, "B1_sync", 0, "conditional")) for rd in mc3})
        RG = R["regression60"]
        both("r60_dwv0_vs_b1", {rd: (cell(RG[rd], "deltawal_versioned", 0), cell(RG[rd], "B1_sync", 0)) for rd in rd3},
             groups=("regression60", "regression60"))
        # robustness runs: reordering and ablations of DeltaWAL-Versioned (lag 1)
        O, A = R["reorder"], R["ablation"]
        for lag in (1, 3):
            both(f"reorder_dwv_lag{lag}_p03_vs_p0", {rd: (cell(O[rd], "deltawal_versioned", lag, 0.3),
                                                          U(rd, "deltawal_versioned", lag)) for rd in rd3})
            both(f"reorder_dwv_lag{lag}_p03_vs_p0_ku", {rd: (ku(cell(O[rd], "deltawal_versioned", lag, 0.3)),
                                                             ku(U(rd, "deltawal_versioned", lag))) for rd in rd3})
        for v in ("nogate", "noprec", "notags", "recency_user"):
            both(f"abl_{v}", {rd: (cell(A[rd], f"deltawal_versioned_{v}", 1), U(rd, "deltawal_versioned", 1))
                              for rd in rd3})
            both(f"abl_{v}_ku", {rd: (ku(cell(A[rd], f"deltawal_versioned_{v}", 1)), ku(U(rd, "deltawal_versioned", 1)))
                                 for rd in rd3})
        both("abl_recency_user_vs_b3", {rd: (cell(A[rd], "deltawal_versioned_recency_user", 1),
                                             H(rd, "B3_recency_matched_user", 1)) for rd in rd3})
        both("abl_recency_user_vs_b3_tr", {rd: (tr(cell(A[rd], "deltawal_versioned_recency_user", 1)),
                                                tr(H(rd, "B3_recency_matched_user", 1))) for rd in rd3})
        B3D = R["b3dated"]
        for nm, sel in (("", lambda rs: rs), ("_ku", ku), ("_tr", tr)):
            both(f"b3dated_vs_b3{nm}", {rd: (sel(cell(B3D[rd], "B3_recency_matched_user_dated", 1)),
                                             sel(H(rd, "B3_recency_matched_user", 1))) for rd in rd3})
            both(f"recency_tags_vs_b3dated{nm}", {rd: (sel(cell(A[rd], "deltawal_versioned_recency_user", 1)),
                                                       sel(cell(B3D[rd], "B3_recency_matched_user_dated", 1)))
                                                  for rd in rd3})
        # relabelled KU-past split (robustness of the cue-word split; see the appendix)
        relab = {"5a4f22c0": False, "031748ae": True, "c6853660": True}
        rl = lambda rs: [r for r in rs if is_ku(r) and relab.get(r["qid"], r["ku_past"])]   # noqa: E731
        both("kupast_relabelled_dwv0_vs_b1", {rd: (rl(U(rd, "deltawal_versioned", 0)), rl(H(rd, "B1_sync", 0)))
                                              for rd in rd3})

    # ---------------------------------------------------------------- Table 10: ablations and reordering
    def tab10(self) -> str:
        A, O, rd3 = self.R["ablation"], self.R["reorder"], ["gemma4", "glm-5.3", "gpt-6-luna"]
        spec = [("Full DeltaWAL-V.", lambda rd: cell(self.R["unified"][rd], "deltawal_versioned", 1)),
                ("No relevance gate (budget-matched tail)", lambda rd: cell(A[rd], "deltawal_versioned_nogate", 1)),
                ("No precedence instruction", lambda rd: cell(A[rd], "deltawal_versioned_noprec", 1)),
                ("No epoch tags (dates kept)", lambda rd: cell(A[rd], "deltawal_versioned_notags", 1)),
                ("Recency tail + tags + instruction", lambda rd: cell(A[rd], "deltawal_versioned_recency_user", 1)),
                ("Full, reordering $p_\\text{ooo}=0.3$", lambda rd: cell(O[rd], "deltawal_versioned", 1, 0.3)),
                None,
                ("B3 (Table~\\ref{tab:main138}): undated recent turns",
                 lambda rd: cell(self.R["heldout"][rd], "B3_recency_matched_user", 1)),
                ("B3 + session dates on recent turns",
                 lambda rd: cell(self.R["b3dated"][rd], "B3_recency_matched_user_dated", 1))]
        body, rec = [], {}
        for item in spec:
            if item is None:
                body.append("\\midrule")
                continue
            label, get = item
            cells = []
            for rd in rd3:
                rs = get(rd)
                assert len(rs) == 138, (label, rd, len(rs))
                st = {"overall": stat(rs), "KU": stat([r for r in rs if is_ku(r)]),
                      "temporal": stat([r for r in rs if is_tr(r)]),
                      "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs), 1)}
                rec.setdefault(label, {})[rd] = st
                cells += [tnum(st["overall"]["mean"]), tnum(st["KU"]["mean"]), tnum(st["temporal"]["mean"])]
            body.append(f"{label} & " + " & ".join(cells) + " \\\\")
        keymap = {"Full DeltaWAL-V.": "full", "No relevance gate (budget-matched tail)": "nogate",
                  "No precedence instruction": "noprec", "No epoch tags (dates kept)": "notags",
                  "Recency tail + tags + instruction": "recency_tags", "Full, reordering $p_\\text{ooo}=0.3$": "reorder",
                  "B3 (Table~\\ref{tab:main138}): undated recent turns": "b3",
                  "B3 + session dates on recent turns": "b3_dated"}
        self.rec("tab10", {keymap[k]: v for k, v in rec.items()})
        hdr = ["Variant (lag 1)"] + [f"\\multicolumn{{3}}{{c}}{{{rd}}}" for rd in rd3]
        sub = " & " + " & ".join(["All", "KU", "TR"] * 3) + " \\\\"
        t = tabular("l" + "rrr" * 3, hdr, [sub, "\\midrule"] + body)
        t = t.replace(" \\\\\n\\midrule\n & All", " \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n & All", 1)
        b = self.s["config"]
        cap = ("Read-path ablations of DeltaWAL-Versioned at lag~1 on the 138 held-out questions (accuracy, \\%; "
               "All, knowledge-update, temporal), each changing one factor, and the full system under reordering. "
               "The no-gate and recency variants fill the same token budget as the gated prompt (mean "
               f"prompt within 1\\% of full DeltaWAL-V.); the recency variant uses the trailing user turns of the "
               f"conversation, as B3, rendered with epoch tags, dates and the precedence instruction. Gate recall "
               f"(an update-session turn passes $\\tau={b['wal_sim_threshold']}$) is 60/60 at lags 1, 3 and 5. "
               f"Bottom: B3 as in Table~\\ref{{tab:main138}} (async in-place store, "
               f"{b['b3_matched_token_budget']}-token budget, undated recent turns) and the same system with only "
               f"session dates added to its recent turns.")
        return table_env("tab:ablation", cap, [t])

    # ---------------------------------------------------------------- Table 11: gemma4 temporal probes
    def tab11(self) -> str:
        T, U = self.R["temporal_probe"]["gemma4"], self.R["unified"]["gemma4"]
        base1 = [r for r in cell(U, "deltawal_versioned", 1) if is_tr(r)]
        spec = [("DeltaWAL-V., lag 0", 0, [r for r in cell(U, "deltawal_versioned", 0) if is_tr(r)]),
                ("DeltaWAL-V., lag 1", 1, base1),
                ("lag 1, uncached re-run", 1, [r for r in cell(T, "deltawal_versioned_rerun", 1) if is_tr(r)]),
                ("lag 0, padded to the lag-1 prompt length", 0, [r for r in cell(T, "deltawal_versioned_pad", 0) if is_tr(r)]),
                ("lag 1, tail dated as ``message sent on''", 1, [r for r in cell(T, "deltawal_versioned_sentdate", 1) if is_tr(r)]),
                ("lag 1, tail placed before the facts", 1, [r for r in cell(T, "deltawal_versioned_tailfirst", 1) if is_tr(r)]),
                ("lag 1, no precedence instruction", 1, [r for r in cell(T, "deltawal_versioned_noprec", 1) if is_tr(r)])]
        body, rec = [], {}
        for label, lag, rs in spec:
            st = {"temporal": stat(rs), "vs_lag1": paired(rs, base1) if rs is not base1 else None}
            rec[label] = st
            vs = "--" if st["vs_lag1"] is None else f"{st['vs_lag1']['a_only']}/{st['vs_lag1']['b_only']}"
            pv = "--" if st["vs_lag1"] is None else f"{st['vs_lag1']['p_mcnemar']:.3f}"
            body.append(f"{label} & {tci(st['temporal'])} & {vs} & {pv} \\\\")
        slugs = ["lag0", "lag1", "rerun", "pad", "sentdate", "tailfirst", "noprec"]
        self.rec("tab11", {sl: rec[label] for sl, (label, _, _) in zip(slugs, spec)})
        cap = ("gemma4 on the 52 temporal questions: prompt probes for the lag-1 drop. Wins/losses and exact McNemar "
               "$p$ are paired against DeltaWAL-V.\\ at lag~1 (judge: qwen3.8-flash).")
        return table_env("tab:temporal_probe", cap, [tabular("lccr", ["Prompt", "Temporal", hdr2("Wins/losses", "vs.\\ lag 1"), "$p$"], body)])

    # ---------------------------------------------------------------- Table 12: judge agreement across families
    def tab12(self) -> str:
        hz = self.s.get("rejudge_haiku") or []
        if not hz:
            return "% no second-judge data\n"
        key = {}
        for g, runs in self.R.items():
            for rd, rows in runs.items():
                for r in rows:
                    key[(g, rd, r["qid"], r["system"], r["lag"], float(r["p_ooo"] or 0.0))] = r

        def kappa(x, y):
            n = len(x)
            po = sum(a == b for a, b in zip(x, y)) / n
            pa, pb = sum(x) / n, sum(y) / n
            pe = pa * pb + (1 - pa) * (1 - pb)
            return po, ((po - pe) / (1 - pe) if pe < 1 else 1.0)
        groups = defaultdict(lambda: {"c": ([], []), "s": ([], [])})
        names = {"heldout": "LongMemEval-S held-out grid", "unified": "DeltaWAL-V.\\ cells", "controls": "Controls",
                 "regression60": "Non-update regression", "mem0": "\\memzero{}", "mem0hist": "\\memzero{} history hook",
                 "memconflict": "MemConflict", "ablation": "Read-path ablations", "b3dated": "B3 + session dates",
                 "reorder": "DeltaWAL-V.\\ under reordering", "temporal_probe": "gemma4 temporal probes"}
        for g, rd, q, s, l, p, c, a, u in hz:
            r = key.get((g, rd, q, s, l, float(p or 0.0)))
            if r is None:
                continue
            for gg in (g, "ALL"):
                groups[gg]["c"][0].append(bool(r["correct"]))
                groups[gg]["c"][1].append(bool(c))
                if r["qtype"] == "knowledge-update" or r["qtype"].startswith("memconflict-static"):
                    groups[gg]["s"][0].append(stale(r))
                    groups[gg]["s"][1].append((not c) and bool(u))
        body, rec = [], {}
        for g in list(names) + ["ALL"]:
            d = groups.get(g)
            if not d:
                continue
            pc, kc = kappa(*d["c"])
            ps, ks = kappa(*d["s"]) if d["s"][0] else (None, None)
            rec[g] = {"n": len(d["c"][0]), "agree_correct_pct": round(100 * pc, 1), "kappa_correct": round(kc, 3),
                      "n_stale": len(d["s"][0]), "agree_stale_pct": round(100 * ps, 1) if ps is not None else None,
                      "kappa_stale": round(ks, 3) if ks is not None else None}
            label = "All judged rows" if g == "ALL" else names[g]
            if g == "ALL":
                body.append("\\midrule")
            body.append(f"{label} & {rec[g]['n']} & {tnum(rec[g]['agree_correct_pct'])} & {rec[g]['kappa_correct']:.2f} & "
                        + (f"{rec[g]['kappa_stale']:.2f}" if rec[g]["kappa_stale"] is not None else "--") + " \\\\")
        # Codex (GPT family) on the 50-row judge-check sheet
        j = self.s.get("judge_check50_codex") or []
        if j:
            jc = kappa([r["judge_correct"] for r in j], [r["codex_correct"] for r in j])
            ja = kappa([r["judge_abstained"] for r in j], [r["codex_abstained"] for r in j])
            kus = [r for r in j if r["stratum"] == "knowledge-update"]
            js = kappa([(not r["judge_correct"]) and r["judge_uses_superseded"] for r in kus],
                       [(not r["codex_correct"]) and r["codex_uses_superseded"] for r in kus])
            rec["codex50"] = {"n": len(j), "agree_correct_pct": round(100 * jc[0], 1), "kappa_correct": round(jc[1], 3),
                              "kappa_abstained": round(ja[1], 3), "n_stale": len(kus), "kappa_stale": round(js[1], 3)}
        self.rec("tab12", rec)
        cap = ("Agreement of the qwen3.8-flash judge with a second judge from another model family (claude-haiku-5-5) "
               "on every judged answer in the paper, with identical judge prompts. Stale = wrong and using the "
               "superseded value (KU rows; static false belief for MemConflict).")
        return table_env("tab:judges", cap, [tabular("lrrrr", ["Rows", "$n$", hdr2("Agree", "correct (\\%)"),
                                                                 hdr2("$\\kappa$", "correct"), hdr2("$\\kappa$", "stale")], body)],
                         wide=False)

    def tab9(self) -> str:
        """Evaluated MemConflict questions that are not clean by both raters (dropped in the clean re-scoring)."""
        a1 = {r["qid"]: r for r in self.s.get("memconflict_audit90") or []}
        a2 = {r["qid"]: r["verdict"] for r in self.s.get("memconflict_audit90_r2") or []}
        if not a1 or not a2:
            return "% audit data missing from the snapshot\n"
        order = {"dynamic": 0, "static": 1, "conditional": 2}
        drop = sorted((q for q in a1 if not (a1[q]["verdict"] == "clean" and a2[q] == "clean")),
                      key=lambda q: (order[a1[q]["type"]], q))
        self.rec("tab9", {"n_dropped": len(drop), "qids": drop})
        short_v = {"clean": "clean", "ambiguous": "ambig.", "label_error": "label err."}
        body = [f"\\texttt{{{tex_escape(q)}}} & {a1[q]['type']} & {short_v[a1[q]['verdict']]} & {short_v[a2[q]]} \\\\"
                for q in drop]
        cap = (f"The {len(drop)} evaluated MemConflict questions that are not rated clean by both raters; they are "
               f"excluded in the clean re-scoring (Table~\\ref{{tab:memconflict_clean}}).")
        return table_env("tab:memconflict_dropped", cap,
                         [tabular("llll", ["Question id", "Type", "Rater 1", "Rater 2"], body)], wide=True)

    def tab6(self) -> str:
        L = self.s["latency"]
        wa, rq, rx, cold = (L["write_ack"], L["read_path_excl_reader"], L["read_path_excl_reader_and_query_embedding"],
                            L["cold_read_excl_reader_and_query_embedding"])
        spec = [("B1 sync", wa["B1_sync"], rq["B1_sync"], rx["B1_sync"]),
                ("B2 async", wa["B2_async"], rq["B2_async"], rx["B2_async"]),
                ("DeltaWAL (warm)", wa["deltawal_*"], rq["deltawal_prompt"], rx["deltawal_prompt"]),
                ("DeltaWAL (cold)", wa["deltawal_*"], None, cold["deltawal_prompt"]),
                ("Shadow v2 (warm)", wa["deltawal_*"], rq["deltawal_shadow_v2"], rx["deltawal_shadow_v2"]),
                ("Shadow v2 (cold)", wa["deltawal_*"], None, cold["deltawal_shadow_v2"]),
                ("Lazy", wa["deltawal_*"], rq["deltawal_lazy"], rx["deltawal_lazy"])]

        def ms(d, k):
            if d is None:
                return "--"
            v = d[k]
            return f"{v:,.0f}" if v >= 1000 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")
        body = []
        for name, w, a, b in spec:
            self.rec(f"tab6/{name}", {"write_ack": w, "read_incl_query_embedding": a, "read_excl_query_embedding": b})
            body.append(f"{name} & {ms(w, 'p50_ms')} & {ms(w, 'p95_ms')} & {ms(a, 'p50_ms')} & {ms(a, 'p95_ms')} & "
                        f"{ms(b, 'p50_ms')} & {ms(b, 'p95_ms')} \\\\")
        self.rec("tab6/query_embedding", L["query_embedding"])
        self.rec("tab6/config", L["config"])
        c = L["config"]
        hdr = ["System", "\\multicolumn{2}{c}{" + hdr2("Write", "ack") + "}",
               "\\multicolumn{2}{c}{" + hdr2("Read path", "incl.\\ emb.") + "}",
               "\\multicolumn{2}{c}{" + hdr2("Read path", "excl.\\ emb.") + "}"]
        xl = self.s.get("extractor_latency") or {}
        if xl:
            self.rec("tab6/cloud_extractor_wall_s", xl)
        sub = " & p50 & p95 & p50 & p95 & p50 & p95 \\\\"
        t = tabular("lrrrrrr", hdr, [sub, "\\midrule"] + body)
        t = t.replace(" \\\\\n\\midrule\n & p50", " \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n & p50", 1)
        cap = (f"Wall-clock latency (ms) on the real stack (Redpanda, Redis, PostgreSQL/pgvector, local "
               f"{tex_escape(c['embedder'])} embeddings), {c['questions']} dev questions at lag {c['lag']}; write ack "
               f"over {wa['B1_sync']['n']} sessions, read paths exclude the reader decode. B1 acknowledges only after "
               f"synchronous extraction (recorded live time of the local 12B extractor on the dev split), embedding and "
               f"upsert; B2 after the broker ack; DeltaWAL after the Redis WAL write-through and broker ack. Cold = the "
               f"WAL tail is embedded at read time; Lazy runs a live extraction call per query; `emb.' = the query "
               f"embedding. DeltaWAL-Versioned uses the DeltaWAL read path."
               + (f" For reference, the cloud extractor used for all evaluated sets took "
                  f"{xl['median_s']:.2f}\\,s per session at the median ({xl['p95_s']:.2f}\\,s p95; {xl['n']:,} live calls "
                  f"logged during concurrent batch extraction)." if xl else ""))
        return table_env("tab:latency", cap, [t])


# ============================================================================================ build: figures
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "purple": "#CC79A7", "vermillion": "#D55E00", "sky": "#56B4E9",
         "green": "#009E73", "yellow": "#F0E442", "black": "#000000"}
SYSCOL = {"B1_sync": OKABE["blue"], "B2_async": OKABE["orange"], "B3_recency_matched_user": OKABE["purple"],
          "deltawal_shadow_v2": OKABE["vermillion"], "deltawal_lazy": OKABE["sky"], "deltawal_versioned": OKABE["green"],
          "Mem0_sync": OKABE["black"], "Mem0_async": OKABE["orange"], "Mem0_deltawal": OKABE["green"],
          "Mem0_sync_history": OKABE["yellow"]}
RMARK = {"gemma4": "o", "glm-5.3": "s", "gpt-6-luna": "^", "glm-5.3-flash": "D"}
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d9d9d9"
COL_W = 6.5


def mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.hashsalt": "epochmem", "legend.frameon": False, "lines.linewidth": 1.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    return plt


class LayoutError(RuntimeError):
    pass


def check_layout(fig, name, boxes=()):
    """Fail on overlapping text, text outside the figure, or a box label that does not fit its box."""
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    from matplotlib.axis import Axis
    ticklabels = {id(t) for a in fig.axes for ax_ in (a.xaxis, a.yaxis)
                  for t in ax_.get_ticklabels(minor=False) + ax_.get_ticklabels(minor=True)}
    drawn = [t.label1 for a in fig.axes if a.axison for ax_ in (a.xaxis, a.yaxis) for t in ax_._update_ticks()
             if t.label1.get_visible() and ax_.get_visible()]
    texts = [t for t in fig.findobj(Text) if id(t) not in ticklabels] + drawn
    seen, items = set(), []
    for t in texts:
        if id(t) in seen or not t.get_visible() or not t.get_text().strip():
            continue
        seen.add(id(t))
        items.append((t, t.get_window_extent(r)))
    fb = fig.bbox
    probs = []
    for t, b in items:
        if b.x0 < fb.x0 - 0.5 or b.y0 < fb.y0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y1 > fb.y1 + 0.5:
            probs.append(f"outside figure: {t.get_text()!r}")
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if a.x0 < b.x1 - 0.5 and b.x0 < a.x1 - 0.5 and a.y0 < b.y1 - 0.5 and b.y0 < a.y1 - 0.5:
                probs.append(f"overlap: {items[i][0].get_text()!r} x {items[j][0].get_text()!r}")
    for patch, text in boxes:
        pb, tb = patch.get_window_extent(r), text.get_window_extent(r)
        if tb.x0 < pb.x0 + 1 or tb.x1 > pb.x1 - 1 or tb.y0 < pb.y0 + 1 or tb.y1 > pb.y1 - 1:
            probs.append(f"label does not fit its box: {text.get_text()!r}")
    if probs:
        raise LayoutError(f"{name}: " + "; ".join(probs))
    return len(items)


def save(fig, name, report):
    out = PAPER / "figures"
    (out / "preview").mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": None,
                                                 "Producer": None})
    fig.savefig(out / "preview" / f"{name}.png", dpi=200, metadata={"Software": None})
    w, h = fig.get_size_inches()
    report[name] = {"width_in": round(float(w), 3), "height_in": round(float(h), 3)}


class Figures:
    def __init__(self, b: Builder, plt):
        self.b, self.plt, self.M, self.report = b, plt, b.M, {}

    # ---------------------------------------------------------------- fig 1: architecture
    def fig1(self):
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
        plt = self.plt
        L, cfg = self.b.s["latency"], self.b.s["config"]
        ack = L["write_ack"]["deltawal_*"]["p50_ms"]
        rd = L["read_path_excl_reader_and_query_embedding"]["deltawal_prompt"]["p50_ms"]
        fig = plt.figure(figsize=(COL_W, 2.9))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 44.6)
        ax.axis("off")
        boxes = []

        def box(x, y, w, h, text, fc="#ffffff", ec=MUTED, bold=False, size=7.6):
            p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=0.9)
            ax.add_patch(p)
            t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                        fontweight="bold" if bold else "normal", color=INK, linespacing=1.25)
            boxes.append((p, t))
            return p

        def arrow(x0, y0, x1, y1, color=MUTED, style="-|>", ls="-"):
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=9, color=color,
                                         lw=1.0, linestyle=ls, shrinkA=0, shrinkB=0))

        green_fc, blue_fc = "#e3f4ee", "#e2eef7"
        ax.text(1, 42.6, "Write path", fontsize=9, fontweight="bold", va="center")
        ax.text(1, 19.6, "Read path", fontsize=9, fontweight="bold", va="center")
        # write path
        box(1, 29, 11, 9, "Client\nsession")
        box(17, 32.5, 22, 7.5, "Redis WAL\nraw turns + epoch $e$", fc=green_fc, ec=OKABE["green"])
        box(17, 23.5, 22, 7, "Redpanda topic\n(async)")
        box(45, 23.5, 17, 7, "Worker:\nfact extraction")
        box(68, 23.5, 31, 9.5, "Append-only PostgreSQL\nkey (entity_key, source_epoch)\nall versions kept",
            fc=green_fc, ec=OKABE["green"])
        arrow(12, 35.5, 17, 36.2)
        arrow(12, 31.5, 17, 27)
        arrow(39, 27, 45, 27)
        arrow(62, 27, 68, 27)
        ax.text(40.5, 38.4, f"ack p50 {ack:.1f} ms\n(WAL + broker ack)", fontsize=7.4, color=OKABE["green"],
                fontweight="bold", va="center", ha="left", linespacing=1.2)
        arrow(80, 33, 39, 34.2, color=MUTED, style="-|>", ls=(0, (2, 2)))
        ax.text(61, 35.6, "commit advances the watermark", fontsize=7, color=MUTED, ha="left", va="center")
        # read path
        box(1, 5, 11, 9, "Query")
        box(17, 8.5, 22, 8.5, "Single Lua snapshot\nwatermark + WAL tail\n($e >$ committed)", fc=blue_fc,
            ec=OKABE["blue"])
        box(45, 8.5, 17, 8.5, f"Relevance gate\ncos $\\geq$ {cfg['wal_sim_threshold']:.2f}\n(top {cfg['wal_top_m']})",
            fc=blue_fc, ec=OKABE["blue"])
        box(45, 0.8, 17, 6, f"Top-{cfg['top_k_facts']} versions")
        box(68, 2.5, 18, 12, "Epoch-ordered\nmerge\n(newer wins;\nolder kept)", fc=blue_fc, ec=OKABE["blue"])
        box(90, 5, 9, 9, "Reader\nLLM")
        arrow(12, 9.5, 17, 12.7)
        arrow(39, 12.7, 45, 12.7)
        arrow(62, 12.7, 68, 11)
        arrow(62, 3.8, 68, 6)
        arrow(86, 8.5, 90, 9.5)
        arrow(12, 7, 45, 3.8)
        ax.text(17, 19.6, f"read path p50 {rd:.1f} ms (excl. reader & query embedding)", fontsize=7.4,
                color=OKABE["blue"], fontweight="bold", va="center", ha="left")
        save(fig, "fig1_architecture", self.report)
        n = check_layout(fig, "fig1_architecture", boxes)
        self.M["fig1"] = {"write_ack_p50_ms": ack, "read_path_p50_ms_excl_reader_and_query_embedding": rd,
                          "wal_sim_threshold": cfg["wal_sim_threshold"], "text_items_checked": n}
        plt.close(fig)

    # ---------------------------------------------------------------- fig 2: pareto
    def fig2(self):
        plt = self.plt
        L = self.b.s["latency"]
        wa, rx = L["write_ack"], L["read_path_excl_reader_and_query_embedding"]
        systems = [("B1_sync", 0, wa["B1_sync"], rx["B1_sync"]), ("B2_async", 1, wa["B2_async"], rx["B2_async"]),
                   ("deltawal_shadow_v2", 1, wa["deltawal_*"], rx["deltawal_shadow_v2"]),
                   ("deltawal_lazy", 1, wa["deltawal_*"], rx["deltawal_lazy"]),
                   ("deltawal_versioned", 1, wa["deltawal_*"], rx["deltawal_prompt"])]
        readers = ["gemma4", "glm-5.3"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.85), constrained_layout=True)
        data = {}
        for system, lag, w, rd in systems:
            for reader in readers:
                st = self.M[f"tab1/{reader}/{system}/lag{lag}"]
                data[(system, reader)] = (w["p50_ms"], rd["p50_ms"], st["KU"]["mean"], st["temporal"]["mean"])
        # left: write ack vs KU
        dodge = {"deltawal_shadow_v2": 0.8, "deltawal_lazy": 1.25}
        for system, lag, w, rd in systems:
            f_ = dodge.get(system, 1.0)
            data.update({(system, r): (data[(system, r)][0] * f_,) + data[(system, r)][1:] for r in readers})
        for system, lag, w, rd in systems:
            xs = [data[(system, r)][0] for r in readers]
            ys = [data[(system, r)][2] for r in readers]
            a1.plot(xs, ys, color=SYSCOL[system], lw=0.8, alpha=0.6, zorder=2)
            for r in readers:
                x, _, y, _ = data[(system, r)]
                a1.scatter([x], [y], s=34, marker=RMARK[r], color=SYSCOL[system], edgecolor="white", linewidth=0.8,
                           zorder=3)
        a1.set_xscale("log")
        a1.set_xlim(0.5, 30000)
        a1.set_ylim(0, 100)
        a1.set_xlabel("Write-ack latency p50 (ms, log)")
        a1.set_ylabel("Accuracy (%)")
        a1.set_title("(a) Write path: KU")
        x_dw = wa["deltawal_*"]["p50_ms"]
        a1.text(x_dw * 1.7, 66, "DeltaWAL variants\n(one shared write path;\noffset for legibility)", fontsize=7.2,
                color=INK, ha="left", va="center", linespacing=1.15)
        a1.text(data[("B1_sync", "gemma4")][0] * 0.8, 70, "B1 sync", fontsize=7.4, ha="right", va="center")
        a1.text(data[("B2_async", "gemma4")][0] * 1.3, 11, "B2 async", fontsize=7.4, ha="left", va="center")
        # right: read path vs KU (filled) and temporal (hollow). B1/B2 (~4 ms) and DeltaWAL/Shadow (~11 ms) have
        # near-identical read latency, so points are offset horizontally (stated in the caption); readers too.
        # offsets keep each pair in its true latency order: B2 (4.09 ms) left of B1 (4.21), DeltaWAL left of Shadow
        sys_dodge = {"B1_sync": 1.15, "B2_async": 0.87, "deltawal_versioned": 0.78, "deltawal_shadow_v2": 1.28}
        rd_dodge = {"gemma4": 0.93, "glm-5.3": 1.07}
        for system, lag, w, rd in systems:
            for r in readers:
                _, x, ku, tr = data[(system, r)]
                x = x * sys_dodge.get(system, 1.0) * rd_dodge[r]
                a2.plot([x, x], [tr, ku], color=SYSCOL[system], lw=0.8, alpha=0.6, zorder=2)
                a2.scatter([x], [ku], s=34, marker=RMARK[r], color=SYSCOL[system], edgecolor="white", linewidth=0.8,
                           zorder=3)
                a2.scatter([x], [tr], s=34, marker=RMARK[r], facecolor="white", edgecolor=SYSCOL[system],
                           linewidth=1.2, zorder=3)
        a2.set_xscale("log")
        a2.set_xlim(2, 20000)
        a2.set_ylim(0, 100)
        a2.set_xlabel("Read-path latency p50 (ms, log; excl. reader)")
        a2.set_title("(b) Read path: KU and temporal")
        gx = data[("deltawal_versioned", "gemma4")][1] * sys_dodge["deltawal_versioned"] * rd_dodge["gemma4"]
        g_tr, l_tr = data[("deltawal_versioned", "gemma4")][3], data[("deltawal_versioned", "glm-5.3")][3]
        a2.annotate(f"12B temporal drop\n(gemma4 TR {g_tr:.0f}% vs glm-5.3 {l_tr:.0f}%)", xy=(gx * 1.08, g_tr),
                    xytext=(150, 40), fontsize=7.2, color=INK, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7))
        # legends: systems (colour) + encodings (marker / fill)
        from matplotlib.lines import Line2D
        sys_h = [Line2D([], [], marker="o", ls="", color=SYSCOL[s], markersize=6, label=SYSNAME[s].replace(" (mutable)", ""))
                 for s, *_ in systems]
        enc_h = [Line2D([], [], marker=RMARK[r], ls="", color=MUTED, markersize=6, label=r) for r in readers] + \
                [Line2D([], [], marker="o", ls="", color=MUTED, markersize=6, label="KU"),
                 Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=MUTED, markersize=6,
                        label="Temporal")]
        a2.legend(handles=enc_h, loc="lower right", fontsize=7.2, handletextpad=0.3, borderaxespad=0.2,
                  labelspacing=0.3)
        fig.legend(handles=sys_h, loc="outside lower center", ncol=5, fontsize=7.4, handletextpad=0.3,
                   columnspacing=1.2)
        save(fig, "fig2_pareto", self.report)
        n = check_layout(fig, "fig2_pareto")
        self.M["fig2"] = {f"{s}/{r}": {"write_ack_p50_ms": v[0] / dodge.get(s, 1.0), "read_p50_ms": v[1], "KU": v[2],
                                       "temporal": v[3]} for (s, r), v in sorted(data.items())}
        self.M["fig2"]["note"] = "DeltaWAL-Versioned read latency = deltawal_prompt read path (same gate + merge)"
        plt.close(fig)

    # ---------------------------------------------------------------- fig 3: destructive symmetry
    def fig3(self):
        import numpy as np
        plt = self.plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.75), constrained_layout=True,
                                     gridspec_kw={"width_ratios": [1.25, 1]})
        groups = [("B1_sync", "heldout", 0, "B1 sync: write overwrite"),
                  ("Mem0_sync", "mem0", 0, "Mem0 sync: write overwrite"),
                  ("deltawal_shadow_v2", "heldout", 1, "Shadow v2: read masking"),
                  ("deltawal_versioned", "unified", 1, "DeltaWAL-Versioned")]
        x = np.arange(len(READERS3))
        w = 0.19
        vals = {}
        for i, (system, grp, lag, label) in enumerate(groups):
            ms, lo, hi = [], [], []
            for reader in READERS3:
                rs = [r for r in cell(self.b.R[grp][reader], system, lag) if is_ku(r) and r["ku_past"]]
                st = stat(rs)
                vals[f"{system}/{reader}"] = st
                ms.append(st["mean"]); lo.append(st["mean"] - st["lo"]); hi.append(st["hi"] - st["mean"])
            a1.bar(x + (i - 1.5) * (w + 0.015), ms, w, color=SYSCOL[system], label=label, edgecolor="white",
                   linewidth=0.6, zorder=2)
            a1.errorbar(x + (i - 1.5) * (w + 0.015), ms, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.7,
                        capsize=1.6, zorder=3)
        n_past = vals[f"B1_sync/gemma4"]["n"]
        a1.set_xticks(x, READERS3)
        a1.set_ylim(0, 112)
        a1.set_yticks([0, 25, 50, 75, 100])
        a1.set_ylabel(f"KU-past accuracy (%, $n$={n_past})")
        a1.set_title("(a) Symmetric history erasure")
        a1.grid(axis="x", visible=False)
        a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=7.2, handlelength=1.0,
                  columnspacing=1.0, handletextpad=0.4)
        # right: sweep
        sw = self.b.shadow_sweep()
        labels = [l.split()[0] for l, _ in sw]
        xs = np.arange(len(sw))
        for key, col, mk, name in (("KU_past", OKABE["vermillion"], "o", "KU-past"),
                                   ("KU_current", OKABE["blue"], "s", "KU-current")):
            m = [d[key]["mean"] for _, d in sw]
            lo = [d[key]["lo"] for _, d in sw]
            hi = [d[key]["hi"] for _, d in sw]
            a2.fill_between(xs, lo, hi, color=col, alpha=0.13, linewidth=0, zorder=1)
            a2.plot(xs, m, color=col, marker=mk, markersize=5, zorder=3, label=name)
        frozen = f"{self.b.s['config']['shadow_v2_threshold']:.2f}"
        fi = labels.index(frozen)
        a2.axvline(fi, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=0)
        a2.text(fi - 0.08, 6, "frozen", fontsize=7, color=MUTED, ha="right", va="bottom")
        a2.set_xticks(xs, [l if l != "off" else "off" for l in labels])
        a2.set_xlabel("Shadow v2 threshold (gemma4)")
        a2.set_ylabel("Accuracy (%)")
        a2.set_ylim(0, 105)
        a2.set_title("(b) Masking sweep")
        a2.legend(loc="lower right", fontsize=7.2)
        a2.grid(axis="x", visible=False)
        save(fig, "fig3_destructive_symmetry", self.report)
        check_layout(fig, "fig3_destructive_symmetry")
        self.M["fig3"] = {"ku_past": vals}
        plt.close(fig)

    # ---------------------------------------------------------------- fig 4: external validation
    def fig4(self):
        import numpy as np
        plt = self.plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL_W, 2.9), constrained_layout=True)
        sysl = [("Mem0_sync", "mem0", 0, "Mem0 sync"), ("Mem0_async", "mem0", 1, "Mem0 async"),
                ("Mem0_deltawal", "mem0", 1, "Mem0 + DeltaWAL"), ("Mem0_sync_history", "mem0hist", 0, "sync + history hook")]
        strata = [("KU-past", lambda r: r["ku_past"]), ("KU-current", lambda r: not r["ku_past"])]
        w, vals = 0.19, {}
        present = {j: [i for i, (system, grp, lag, _) in enumerate(sysl)
                       if any(f(r) for rd in READERS3 for r in cell(self.b.R[grp][rd], system, lag))]
                   for j, (_, f) in enumerate(strata)}               # center each group on the bars it has
        for i, (system, grp, lag, label) in enumerate(sysl):
            for j, (sname, f) in enumerate(strata):
                pooled = [dict(r, qid=r["qid"]) for rd in READERS3 for r in cell(self.b.R[grp][rd], system, lag) if f(r)]
                if not pooled:
                    continue
                st = stat(pooled, cluster=True)
                vals[f"{system}/{sname}"] = st
                k = present[j].index(i)
                xpos = j + (k - (len(present[j]) - 1) / 2) * (w + 0.015)
                a1.bar([xpos], [st["mean"]], w, color=SYSCOL[system], edgecolor=INK if system == "Mem0_sync_history"
                       else "white", linewidth=0.6, label=label if j == 0 else None, zorder=2,
                       hatch="////" if system == "Mem0_deltawal" else None)
                if st["mean"] < 3:                           # a zero-height bar is otherwise invisible
                    a1.text(xpos, 3.5, f"{st['mean']:.0f}", ha="center", va="bottom", fontsize=7.2, color=INK)
                for rd in READERS3:
                    rr = [r for r in cell(self.b.R[grp][rd], system, lag) if f(r)]
                    if not rr:
                        continue
                    a1.scatter([xpos], [100 * sum(r["correct"] for r in rr) / len(rr)], s=13, marker=RMARK[rd],
                               facecolor="white", edgecolor=INK, linewidth=0.7, zorder=4)
        a1.set_xticks([0, 1], [s for s, _ in strata])
        a1.set_ylim(0, 112)
        a1.set_yticks([0, 25, 50, 75, 100])
        a1.set_ylabel("Accuracy (%, pooled over 3 readers)")
        a1.set_title("(a) Mem0 OSS on held-out KU")
        a1.grid(axis="x", visible=False)
        from matplotlib.lines import Line2D
        h, l = a1.get_legend_handles_labels()
        h += [Line2D([], [], marker=RMARK[r], ls="", markerfacecolor="white", markeredgecolor=INK, markersize=4.5)
              for r in READERS3]
        l += READERS3
        a1.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=6.9, handlelength=1.0,
                  columnspacing=0.8, handletextpad=0.3)
        # right: memconflict, pooled over the three readers (per-reader values: Table 5)
        spec = [("B1_sync", 0), ("deltawal_versioned", 0), ("B2_async", 1), ("B3_recency_matched_user", 1),
                ("deltawal_versioned", 1)]
        mc_readers = ["gemma4", "gpt-6-luna", "glm-5.3-flash"]
        names = {("B1_sync", 0): ("B1 sync (lag 0)", (3, -6), "left"),
                 ("deltawal_versioned", 0): ("DeltaWAL-V. lag 0", (3, 7), "left"),
                 ("B2_async", 1): ("B2 async", (3, -6), "left"),
                 ("B3_recency_matched_user", 1): ("B3 recency", (3, 6), "left"),
                 ("deltawal_versioned", 1): ("DeltaWAL-V. lag 1", (-6, 7), "right")}
        pts = {}
        keep = self.b._clean70()                    # the clean subset that Section 5.3 reports
        for system, lag in spec:
            rows = [r for rd in mc_readers for r in cell(self.b.R["memconflict"][rd], system, lag) if r["qid"] in keep]
            sx = stat([r for r in rows if r["qtype"] == "memconflict-static"], f=stale, cluster=True)
            sy = stat([r for r in rows if r["qtype"] == "memconflict-dynamic"], cluster=True)
            pts[(system, lag)] = (sx, sy)
            col = SYSCOL[system]
            a2.errorbar([sx["mean"]], [sy["mean"]], xerr=[[sx["mean"] - sx["lo"]], [sx["hi"] - sx["mean"]]],
                        yerr=[[sy["mean"] - sy["lo"]], [sy["hi"] - sy["mean"]]], fmt="none", ecolor=col,
                        elinewidth=0.8, alpha=0.7, zorder=2)
            if lag == 1:
                a2.scatter([sx["mean"]], [sy["mean"]], s=46, marker="o", color=col, edgecolor="white", linewidth=0.8,
                           zorder=3)
            else:
                a2.scatter([sx["mean"]], [sy["mean"]], s=46, marker="o", facecolor="white", edgecolor=col,
                           linewidth=1.4, zorder=3)
            name, (dx, dy), ha = names[(system, lag)]
            a2.text(sx["mean"] + dx, sy["mean"] + dy, name, fontsize=7.4, ha=ha, va="center", color=INK)
        a2.set_xlim(-5, 100)
        a2.set_ylim(0, 80)
        a2.set_xlabel("Static: believed the later false statement (%)")
        a2.set_ylabel("Dynamic accuracy (%)")
        a2.set_title("(b) MemConflict (pooled readers)")
        a2.text(98, 3, "hollow = lag 0, filled = lag 1", fontsize=7, color=MUTED, ha="right", va="bottom")
        save(fig, "fig4_external", self.report)
        check_layout(fig, "fig4_external")
        self.M["fig4"] = {"mem0_pooled": vals,
                          "memconflict_pooled": {f"{s}/lag{l}": {"static_false_belief": v[0], "dynamic": v[1]}
                                                 for (s, l), v in sorted(pts.items())}}
        plt.close(fig)


# ============================================================================================ main
def build(snap: dict) -> dict:
    b = Builder(snap)
    tdir = PAPER / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("tab1_main_138", b.tab1), ("tab2_history_ablation", b.tab2), ("tab3_regression_60", b.tab3),
                     ("tab4_mem0_60", b.tab4), ("tab5_memconflict_90", b.tab5), ("tab6_latency", b.tab6),
                     ("tab7_memconflict_clean", b.tab7), ("tab8_memconflict_audit", b.tab8),
                     ("tab9_memconflict_dropped", b.tab9), ("tab10_ablation", b.tab10),
                     ("tab11_temporal_probe", b.tab11), ("tab12_judges", b.tab12)):
        (tdir / f"{name}.tex").write_text(fn(), encoding="utf-8", newline="\n")
    b.tests()
    L = snap["latency_durable"]
    b.rec("latency_durable", {k: {s.replace("deltawal_*", "DeltaWAL"): {"p50_ms": round(v["p50_ms"], 2), "p95_ms": round(v["p95_ms"], 2)}
                                  for s, v in L[k]["write_ack"].items()} for k in ("nondurable_rerun", "durable")})
    b.rec("latency_durable/deltawal_added_p50_ms", round(L["durable"]["write_ack"]["deltawal_*"]["p50_ms"]
                                                         - L["nondurable_rerun"]["write_ack"]["deltawal_*"]["p50_ms"], 1))
    fd = snap["forensics"]["destructive_overwrite_qids"]
    b.rec("forensics_distinct", {"pairs": sum(len(v) for v in fd.values()),
                                 "questions": len(set().union(*map(set, fd.values())))})
    plt = mpl_setup()
    F = Figures(b, plt)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        F.fig1(); F.fig2(); F.fig3(); F.fig4()
    if caught:
        raise LayoutError("matplotlib warnings: " + "; ".join(str(w.message) for w in caught))
    ja = snap["judge_agreement"]
    b.rec("judge_agreement_qwen3.8-flash_vs_qwen4-max", {
        k: ja[k] for k in ("n_pairs", "n_questions")} | {
        m: {"kappa": ja[m]["kappa"], "kappa_ci95": ja[m]["kappa_ci95"], "agreement": ja[m]["agreement"], "n": ja[m]["n"]}
        for m in ("correct", "abstained", "stale_read", "confident_wrong")})
    b.rec("datasets/heldout138", b.counts_heldout())
    b.rec("datasets/mem0_ku60", {"n": 60, "KU_past": sum(r["ku_past"] for r in cell(b.R["mem0"]["gemma4"], "Mem0_sync", 0))})
    b.rec("forensics", snap["forensics"]["summary"])
    b.rec("frozen_config", snap["config"])
    b.rec("figures", F.report)
    b.rec("stats", {"bootstrap_resamples": 2000, "ci_level_pct": 95, "seed": snap["config"]["seed"]})
    manifest = {"generator": "scripts/build_paper_assets.py",
                "ci": "95% bootstrap, 2000 resamples of questions, seed 42; pooled cells resample question clusters",
                "units": "accuracy and rates in percent; latency in ms",
                "source_sha256": snap["source_sha256"],
                "metrics": dict(sorted(b.M.items()))}
    (PAPER / "numbers_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=False, ensure_ascii=False) + "\n",
                                                encoding="utf-8", newline="\n")
    write_macros(manifest["metrics"])
    write_appendix_prompts()
    return manifest


def write_appendix_prompts() -> None:
    """paper/tables/app_prompts.tex: the exact prompts, rendered from the code (placeholders in <angle brackets>)."""
    import textwrap
    from types import SimpleNamespace

    import yaml
    from epochmem.eval import judge as J
    from epochmem.memory.extract import SYSTEM_PROMPT
    from epochmem.systems.base import READER_SYSTEM, MemorySystem
    from epochmem.systems.deltawal import PRECEDENCE, _DeltaWALBase
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_memconflict_audit90_sheet import AUDIT_RUBRIC
    from report_controls import PAST as PAST_RE

    cfg = yaml.safe_load((ROOT / "configs/m2_heldout.yaml").read_text(encoding="utf-8"))

    def prep(qtype, abstention=False):
        inst = SimpleNamespace(qtype=qtype, question="<question>", answer="<gold answer>", is_abstention=abstention,
                               question_date="<question date>", meta={})
        return SimpleNamespace(inst=inst, superseded_evidence=lambda *a, **k: ["<superseded evidence turn(s)>"])

    stub = SimpleNamespace(question_block=MemorySystem.question_block, raw_turn=lambda r, c: f"[{r}] {c}")
    fact = {"source_epoch": "<e>", "session_date": "<session date>", "entity_key": "<entity_key>", "fact_text": "<fact>"}
    turn = {"epoch": "<e'>", "date": "<session date>", "role": "<user|assistant>", "content": "<raw turn text>"}
    msgs = _DeltaWALBase._messages(stub, prep("knowledge-update"),
                                   [_DeltaWALBase._fact_line(stub, fact), _DeltaWALBase._turn_line(stub, turn)])
    reader = f"[system]\n{msgs[0]['content']}\n\n[user]\n{msgs[1]['content']}"
    extractor = (f"[system]\n{SYSTEM_PROMPT.format(max_facts=cfg['memory']['max_facts_per_session'])}\n\n[user]\n"
                 f"EXISTING KEYS:\n<up to {cfg['memory']['registry_max_keys']} existing keys, most recent first>\n\n"
                 f"<session date and turns; assistant turns truncated to "
                 f"{cfg['memory']['assistant_trunc_chars_extract']} characters>")
    judge_ku = J.correctness_prompt(prep("knowledge-update"), "<model response>")
    judge_tr = J.correctness_prompt(prep("temporal-reasoning"), "<model response>")
    judge_abs = J.correctness_prompt(prep("knowledge-update", True), "<model response>")
    behavior = J.behavior_prompt(prep("knowledge-update"), "<model response>")
    cues = re.sub(r"\\b|\(|\)", "", PAST_RE.pattern).split("|")

    def block(title, text, note=""):
        lines = []
        for para in text.split("\n"):
            lines += textwrap.wrap(para, 96, subsequent_indent="  ", break_long_words=True) or [""]
        body = "\n".join(lines)
        assert "\\end{verbatim}" not in body
        return (f"\\paragraph{{{title}}}{(' ' + note) if note else ''}\n\\begin{{footnotesize}}\n\\begin{{verbatim}}\n"
                f"{body}\n\\end{{verbatim}}\n\\end{{footnotesize}}\n")

    out = ["% generated by scripts/build_paper_assets.py from the code -- do not edit by hand",
           block("Reader prompt (DeltaWAL variants).", reader,
                 "Baselines use the same system prompt without the precedence paragraph and list facts as "
                 "\\texttt{- (date) key: fact}; at lag 0 DeltaWAL-Versioned has no \\texttt{NEWER MESSAGE} lines."),
           block("Fact extractor.", extractor),
           block("Judge, correctness (knowledge-update rubric).", judge_ku,
                 "Other types use the LongMemEval rubric of their type; temporal adds the off-by-one clause shown "
                 "next, abstention uses the third prompt."),
           block("Judge, temporal clause.", judge_tr.split("\n\n")[0]),
           block("Judge, abstention.", judge_abs.split("\n\n")[0]),
           block("Judge, behaviour (abstention and stale read).", behavior),
           f"\\paragraph{{KU-past cue words.}} A knowledge-update question counts as KU-past if its text matches "
           f"(case-insensitive, whole words) any of: " + ", ".join(f"\\textit{{{c}}}" for c in cues) + ".\n",
           block("MemConflict audit rubric (blind raters).", AUDIT_RUBRIC)]
    (PAPER / "tables" / "app_prompts.tex").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")


# ============================================================================================ prose macros
_KEY_BAD = re.compile(r"[^A-Za-z0-9_./-]+")


def _flatten(prefix: str, v, out: dict) -> None:
    if isinstance(v, dict):
        if {"mean", "lo", "hi"} <= set(v):            # a CI statistic: the bare key means the point estimate
            out[prefix] = v["mean"]
        for k, x in v.items():
            _flatten(f"{prefix}/{k}" if prefix else str(k), x, out)
    elif isinstance(v, list):
        for i, x in enumerate(v):
            _flatten(f"{prefix}/{i}", x, out)
    elif isinstance(v, bool) or v is None:
        return
    elif isinstance(v, (int, float, str)):
        out[prefix] = v


def _fmt_value(key: str, v) -> str:
    if isinstance(v, str):
        return tex_escape(v)
    if isinstance(v, int):
        return f"{v:,}".replace(",", "{,}") if abs(v) >= 10000 else str(v)
    leaf = key.rsplit("/", 1)[-1]
    if leaf in ("p_perm", "p_mcnemar"):         # p-values: three decimals, floor at 0.001
        return "{<}0.001" if v < 0.001 else f"{v:.3f}"
    if "kappa" in key or leaf == "agreement":
        return f"{v:.3f}" if "kappa_ci95" not in key else f"{v:.2f}"
    if key.startswith(("tab6/", "fig1/")) or leaf.endswith("_ms"):
        return f"{v:,.0f}".replace(",", "{,}") if v >= 1000 else (f"{v:.1f}" if v >= 10 else f"{v:.2f}")
    if leaf == "mean_prompt_tokens":
        return f"{v:.0f}"                       # integers, as in the tables
    if leaf in ("wal_sim_threshold", "shadow_v2_threshold"):
        return f"{v:.2f}"
    if leaf in ("width_in", "height_in"):
        return f"{v:g}"
    return f"{v:.1f}"


def write_macros(metrics: dict) -> None:
    """paper/numbers.tex: every manifest value as a LaTeX lookup, so prose never re-types a number.
      \\nm{key}          formatted value (CI statistics: the bare key is the point estimate)
      \\nmci{key}        '<lo>--<hi>' of a CI statistic
      \\nmraw{key}       full-precision value (expandable; for \\fpeval)
      \\nmdiff{a}{b}     a - b, rounded to one decimal (percentage points)
    An unknown key is a hard LaTeX error."""
    flat: dict = {}
    for k, v in metrics.items():
        _flatten(k, v, flat)
    lines = ["% generated by scripts/build_paper_assets.py from paper/numbers_manifest.json -- do not edit by hand",
             "% usage: \\nm{key}, \\nmci{key}, \\nmraw{key}, \\nmdiff{key_a}{key_b}; keys are listed below",
             "\\makeatletter",
             "\\newcommand{\\nm}[1]{\\ifcsname nm@#1\\endcsname\\csname nm@#1\\endcsname"
             "\\else\\PackageError{numbers}{Unknown manifest key: #1}{See paper/numbers.tex}\\fi}",
             "\\newcommand{\\nmraw}[1]{\\csname nmraw@#1\\endcsname}",
             "\\newcommand{\\nmci}[1]{\\nm{#1/lo}--\\nm{#1/hi}}",
             "\\ExplSyntaxOn",
             "\\fp_new:N \\l__nm_diff_fp",
             "\\NewDocumentCommand{\\nmdiff}{mm}{",
             "  \\cs_if_exist:cF {nmraw@#1} {\\PackageError{numbers}{Unknown~manifest~key:~#1}{}}",
             "  \\cs_if_exist:cF {nmraw@#2} {\\PackageError{numbers}{Unknown~manifest~key:~#2}{}}",
             "  \\fp_set:Nn \\l__nm_diff_fp {round(\\use:c{nmraw@#1}-\\use:c{nmraw@#2},1)}",
             "  \\fp_compare:nNnTF {\\l__nm_diff_fp} = {round(\\l__nm_diff_fp,0)}",
             "    {\\fp_use:N \\l__nm_diff_fp .0} {\\fp_use:N \\l__nm_diff_fp}}",
             "\\ExplSyntaxOff"]
    seen = {}
    for key in sorted(flat):
        k = "/".join(_KEY_BAD.sub("_", seg).strip("_") for seg in key.split("/"))
        assert k not in seen, (k, key, seen.get(k))
        seen[k] = key
        v = flat[key]
        lines.append(f"\\expandafter\\def\\csname nm@{k}\\endcsname{{{_fmt_value(key, v)}}}")
        if isinstance(v, (int, float)):
            lines.append(f"\\expandafter\\def\\csname nmraw@{k}\\endcsname{{{v!r}}}")
    lines.append("\\makeatother")
    (PAPER / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-snapshot", action="store_true", help="skip collect; build from paper/data/snapshot.json.gz")
    args = ap.parse_args()
    if not args.from_snapshot:
        if not RUNS.exists():
            raise SystemExit("runs/ not found: use --from-snapshot")
        write_snapshot(collect())
        print(f"wrote {SNAP.relative_to(ROOT)} ({SNAP.stat().st_size / 1024:.0f} KiB)")
    m = build(read_snapshot())
    n_tab = len(list((PAPER / "tables").glob("tab*.tex")))
    print(f"wrote {n_tab} tables + prompts appendix, 4 figures (+ previews), numbers.tex, numbers_manifest.json "
          f"({len(m['metrics'])} metric groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
