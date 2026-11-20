"""Calibrate wal_sim_threshold and shadow_threshold on the 20-question dev split.

Runs the real DeltaWAL write path (WAL + Kafka + CAS store) for every (question, lag, p_ooo)
cell, then evaluates read-path decisions WITHOUT any reader/judge calls. Dataset `has_answer`
labels are used only to score thresholds here; they never enter a prompt.

Gate metrics (per wal_sim_threshold):
  upd_recall  : cells whose latest evidence session is uncommitted where >=1 of its has_answer
                turns passes the gate (and survives top_m)
  noise_turns : mean gated turns per cell that are NOT has_answer turns of any evidence session
  abs_noise   : mean gated turns per abstention cell (all are distractors by construction)
Shadow metrics (per shadow_threshold, at a given gate):
  tp          : KU cells (update uncommitted & gated) where a retrieved fact from an older
                evidence session is masked
  fp_per_cell : mean masked facts NOT from an older evidence session
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from epochmem.config import load_config, repo_path
from epochmem.guard import set_allow_cloud
from epochmem.runner import build_ctx, select_instances
from epochmem.memory.wal_embed import embed_records
from epochmem.systems.deltawal import DeltaWALShadow, gate_wal, shadow_mask, shadow_mask_v2

GATES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
SHADOWS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SHADOWS_V2 = [0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_dev_local.yaml")
    args = ap.parse_args()
    set_allow_cloud(False)
    cfg = load_config(args.config)
    ctx, _ = build_ctx(cfg)
    top_m, k = cfg["systems"]["b4_delta_top_m"], cfg["memory"]["top_k_facts"]
    cells = []
    for prep in select_instances(cfg, None):
        ans = {(s.epoch, ti) for s in prep.stream if s.is_evidence for ti, t in enumerate(s.session.turns) if t.has_answer}
        ev = prep.evidence_epochs()
        latest, older = (ev[-1] if ev else None), set(ev[:-1])
        sysobj = DeltaWALShadow(ctx)
        qemb = ctx.embed_query(prep.inst.question)
        for p in cfg["grid"]["p_ooo"]:
            for lag in cfg["grid"]["lags"]:
                ns = f"calib:{prep.inst.qid}:{lag}:{p}"
                sysobj.ingest(prep, ns, lag, float(p))
                facts = ctx.store.search(ns, qemb, k, with_embedding=True)
                _, delta = ctx.wal.snapshot(ns)
                missing = [t for t in delta if t.get("embedding") is None]
                embed_records(ctx.embed_docs, missing, cfg["systems"], cfg["models"]["embed_dim"])
                cells.append(dict(qid=prep.inst.qid, qtype=prep.inst.qtype, abs=prep.inst.is_abstention, lag=lag, p=p,
                                  ans=ans, latest=latest, older=older, facts=facts, delta=delta, qemb=qemb,
                                  upd_uncommitted=latest is not None and any(t["epoch"] == latest for t in delta)))
                ctx.store.reset(ns)
                ctx.wal.reset(ns)
        print(f"ingested {prep.inst.qid}", flush=True)

    # similarity distributions (diagnostic)
    s_ans, s_other = [], []
    for c in cells:
        for t in c["delta"]:
            sim = float(np.dot(t["embedding"], c["qemb"]) / (np.linalg.norm(t["embedding"]) * np.linalg.norm(c["qemb"])))
            (s_ans if (t["epoch"], t["turn"]) in c["ans"] else s_other).append(sim)
    q = lambda xs: [round(float(np.percentile(xs, x)), 3) for x in (10, 50, 90)] if xs else []
    report = {"query_sim_answer_turns_p10_p50_p90": q(s_ans), "query_sim_other_uncommitted_p10_p50_p90": q(s_other),
              "n_cells": len(cells), "n_upd_uncommitted": sum(c["upd_uncommitted"] for c in cells), "gate": [], "shadow": {}}

    for g in GATES:
        rec = n_rec = 0
        noise, abs_noise = [], []
        for c in cells:
            gated = gate_wal(c["qemb"], c["delta"], g, top_m)
            keys = {(t["epoch"], t["turn"]) for t in gated}
            if c["upd_uncommitted"] and not c["abs"]:
                n_rec += 1
                rec += any(e == c["latest"] for (e, _) in keys & c["ans"])
            noise.append(len(keys - c["ans"]))
            if c["abs"]:
                abs_noise.append(len(keys))
        report["gate"].append({"wal_sim_threshold": g, "upd_recall": f"{rec}/{n_rec}",
                               "noise_turns": round(float(np.mean(noise)), 2),
                               "abs_noise": round(float(np.mean(abs_noise)), 2) if abs_noise else None})
        shadow_rows = []
        for st in SHADOWS:
            tp = n_tp = 0
            fp = []
            for c in cells:
                gated = gate_wal(c["qemb"], c["delta"], g, top_m)
                _, sup = shadow_mask(c["facts"], gated, st)
                fp.append(sum(1 for f in sup if f["source_epoch"] not in c["older"]))
                if c["qtype"] == "knowledge-update" and not c["abs"] and c["upd_uncommitted"] \
                        and any(t["epoch"] == c["latest"] for t in gated) \
                        and any(f["source_epoch"] in c["older"] for f in c["facts"]):
                    n_tp += 1
                    tp += any(f["source_epoch"] in c["older"] for f in sup)
            shadow_rows.append({"shadow_threshold": st, "tp": f"{tp}/{n_tp}", "fp_per_cell": round(float(np.mean(fp)), 3)})
        report["shadow"][str(g)] = shadow_rows

    # Shadow v2 (sentence-level) at the frozen gate
    g = cfg["systems"]["wal_sim_threshold"]
    v2 = []
    for st in SHADOWS_V2:
        tp = n_tp = 0
        fp, fp_cells = [], 0
        for c in cells:
            gated = gate_wal(c["qemb"], c["delta"], g, top_m)
            _, sup = shadow_mask_v2(c["facts"], gated, st)
            bad = sum(1 for f in sup if f["source_epoch"] not in c["older"])
            fp.append(bad)
            fp_cells += int(bad > 0)
            if c["qtype"] == "knowledge-update" and not c["abs"] and c["upd_uncommitted"]                     and any(t["epoch"] == c["latest"] for t in gated)                     and any(f["source_epoch"] in c["older"] for f in c["facts"]):
                n_tp += 1
                tp += any(f["source_epoch"] in c["older"] for f in sup)
        v2.append({"shadow_v2_threshold": st, "tp": f"{tp}/{n_tp}", "fp_per_cell": round(float(np.mean(fp)), 3),
                   "cells_with_fp": f"{fp_cells}/{len(cells)}"})
    report["shadow_v2_at_gate"] = {"gate": g, "rows": v2}

    out = repo_path(f"runs/calibration_{cfg.get('calibration_tag', 'm2_dev')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "shadow"}, indent=1))
    for g, rows in report["shadow"].items():
        print(f"gate={g}: " + "  ".join(f"st={r['shadow_threshold']}: tp {r['tp']} fp/cell {r['fp_per_cell']}" for r in rows))
    print("SHADOW V2 (sentence-level) at gate", report["shadow_v2_at_gate"]["gate"])
    for r in report["shadow_v2_at_gate"]["rows"]:
        print("  ", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
