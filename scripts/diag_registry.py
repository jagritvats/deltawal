"""Diagnostic: how sensitive is update-key collision to the extractor's key registry?

M1 extracts every session once, chronologically, with a registry of keys from ALL earlier
sessions (idealized: as if every earlier session were already committed). A real async
worker only sees keys that are committed when it starts extracting. This script re-extracts
the evidence sessions of every knowledge-update question under committed-only registries:

  chrono      : M1 setting (keys from all sessions < t)
  commit_ooo  : keys from sessions committed before t in the p_ooo commit order
  commit_ooo_w: as commit_ooo, but the last W commits before t are still being extracted
                concurrently (W parallel workers), so their keys are not visible yet

Facts of non-evidence sessions are taken from the cached M1 extraction (first-order approx).
"""
from __future__ import annotations

import argparse
import json

from epochmem.config import load_config
from epochmem.guard import set_allow_cloud
from epochmem.memory.extract import SYSTEM_PROMPT, FACT_SCHEMA, Fact, normalize_key, render_session
from epochmem.runner import build_ctx, select_instances
from epochmem.stream.schedule import commit_order


def extract_with_registry(ctx, ss, keys: list[str]) -> list[Fact]:
    ex = ctx.extractor
    user_msg = ("EXISTING KEYS:\n" + ("\n".join(sorted(keys)) if keys else "(none)") + "\n\n"
                + render_session(ss, ex.user_chars, ex.asst_chars))
    res, _ = ctx.client.chat_json(ex.model, [{"role": "system", "content": SYSTEM_PROMPT.format(max_facts=ex.max_facts)},
                                             {"role": "user", "content": user_msg}], schema=FACT_SCHEMA, max_tokens=1200)
    items = (res.parsed or {}).get("facts", []) if isinstance(res.parsed, dict) else []
    return [Fact(normalize_key(str(i.get("entity_key", ""))), str(i.get("fact", "")), ss.epoch, "unknown", ss.session.date)
            for i in items[: ex.max_facts] if isinstance(i, dict) and str(i.get("fact", "")).strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_local.yaml")
    ap.add_argument("--p-ooo", type=float, default=0.30)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    set_allow_cloud(False)
    cfg = load_config(args.config)
    ctx, _ = build_ctx(cfg)
    cap = cfg["memory"]["registry_max_keys"]
    out = []
    for prep in select_instances(cfg, None):
        if prep.inst.qtype != "knowledge-update" or prep.inst.is_abstention:
            continue
        base = ctx.extractor.extract_instance(prep)          # cached M1 facts
        ev = prep.evidence_epochs()
        order = commit_order(prep.n, args.p_ooo, f"{cfg['seed']}:{prep.inst.qid}:{args.p_ooo}")
        pos = {prep.stream[i].epoch: k for k, i in enumerate(order)}   # epoch -> commit position
        rec = {"qid": prep.inst.qid, "evidence_epochs": ev}
        for variant in ("chrono", "commit_ooo", "commit_ooo_w"):
            keys_by_ev = {}
            for e in ev:
                if variant == "chrono":
                    visible = [x for x in range(1, e)]
                else:
                    lag_w = args.workers if variant == "commit_ooo_w" else 0
                    visible = [x for x in pos if pos[x] < pos[e] - lag_w]
                reg: dict[str, int] = {}
                for x in sorted(visible, key=lambda x: pos[x] if variant != "chrono" else x):
                    for f in base.get(x, []):
                        reg[f.entity_key] = x
                keys = sorted(reg, key=lambda k: -reg[k])[:cap]
                facts = base[e] if variant == "chrono" else extract_with_registry(ctx, prep.stream[e - 1], keys)
                keys_by_ev[e] = {f.entity_key for f in facts}
                rec.setdefault(f"{variant}_future_keys_visible", 0)
                rec[f"{variant}_future_keys_visible"] += sum(1 for x in visible if x > e)
            new = keys_by_ev[ev[-1]]
            old = set().union(*(keys_by_ev[e] for e in ev[:-1]))
            rec[variant] = bool(new & old)
            rec[f"{variant}_update_before_old_in_commit_order"] = pos[ev[-1]] < min(pos[e] for e in ev[:-1])
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items()}), flush=True)
    for v in ("chrono", "commit_ooo", "commit_ooo_w"):
        print(f"{v:14s} update-key collision: {sum(r[v] for r in out)}/{len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
