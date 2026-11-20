"""2x2 Format Authority Probe.

For each knowledge-update dev question, the reader sees ONLY the old value (Day 1) and the new
value (Day 30), each rendered either as a structured memory fact or as a raw conversation turn,
in both orders (old-first / new-first). No precedence hint, only neutral dates. Measures whether
a reader's choice depends on format rather than recency.

  structured item : the extracted fact (cached chronological extraction) of that evidence session
                    most similar to the question
  raw item        : the has_answer turn of that evidence session most similar to the question
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from epochmem.config import load_config, repo_path
from epochmem.guard import set_allow_cloud
from epochmem.runner import build_ctx, select_instances
from epochmem.systems.base import READER_SYSTEM, trunc

FORMATS = [("structured", "raw"), ("raw", "structured"), ("structured", "structured"), ("raw", "raw")]


def cos(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_dev_local.yaml")
    ap.add_argument("--readers", nargs="+", default=["gemma4:12b-it-qat", "opencode-go/glm-5.3", "opencode-go/gpt-5.6-luna"])
    ap.add_argument("--judge", default="opencode-go/qwen4-max")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    cfg["models"]["judge"] = args.judge
    cfg["models"]["reader"] = next((r for r in args.readers if r.startswith("opencode-go/")), args.readers[0])
    ctx, judge = build_ctx(cfg, session_id="ses_epochmem_probe_format")
    ulim = cfg["systems"]["raw_turn_user_chars"]

    items = []
    for prep in select_instances(cfg, None):
        if prep.inst.qtype != "knowledge-update" or prep.inst.is_abstention:
            continue
        facts = ctx.extractor.extract_instance(prep)          # cached chronological pass
        ev = [s for s in prep.stream if s.is_evidence]
        old_s, new_s = ev[0], ev[-1]
        q = ctx.embed_query(prep.inst.question)

        def best_fact(ss):
            fs = facts.get(ss.epoch, [])
            embs = ctx.embed_docs([f.doc_text() for f in fs])
            f = fs[int(np.argmax([cos(q, e) for e in embs]))]
            return f"{f.entity_key}: {f.fact_text}"

        def best_raw(ss):
            cands = [t for t in ss.session.turns if t.has_answer] or [t for t in ss.session.turns if t.role == "user"]
            embs = ctx.embed_docs([trunc(t.content, ulim) for t in cands])
            t = cands[int(np.argmax([cos(q, e) for e in embs]))]
            return f"[{t.role}] {trunc(t.content, ulim)}"

        items.append({"prep": prep, "old": {"structured": best_fact(old_s), "raw": best_raw(old_s)},
                      "new": {"structured": best_fact(new_s), "raw": best_raw(new_s)}})
    print(f"{len(items)} KU questions", flush=True)

    jobs = []
    for reader in args.readers:
        for it in items:
            for fo, fn in FORMATS:
                for order in ("old_first", "new_first"):
                    jobs.append((reader, it, fo, fn, order))

    def run(job):
        reader, it, fo, fn, order = job
        old = f"- (Day 1) {it['old'][fo]}"
        new = f"- (Day 30) {it['new'][fn]}"
        mem = "\n".join([old, new] if order == "old_first" else [new, old])
        msgs = [{"role": "system", "content": READER_SYSTEM},
                {"role": "user", "content": f"Memory:\n{mem}\n\nCurrent date: Day 31\nQuestion: {it['prep'].inst.question}\nAnswer:"}]
        res = ctx.client.chat(reader, msgs, max_tokens=256)
        v = judge.judge(it["prep"], res.content.strip())
        return {"reader": reader, "qid": it["prep"].inst.qid, "old_fmt": fo, "new_fmt": fn, "order": order,
                "answer": res.content.strip(), "correct": v.correct, "abstained": v.abstained,
                "stale": (not v.correct) and v.uses_superseded}

    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(run, jobs))
    out = repo_path("runs/probe_format")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg = defaultdict(list)
    for r in rows:
        agg[(r["reader"], r["old_fmt"], r["new_fmt"])].append(r)
        agg[(r["reader"], r["old_fmt"], r["new_fmt"], r["order"])].append(r)
    lines = [f"{'reader':<28}{'old -> new':<26}{'n':>4}{'acc':>8}{'stale':>8}{'abst':>7}{'acc(old1st)':>13}{'acc(new1st)':>13}"]
    for reader in args.readers:
        for fo, fn in FORMATS:
            rs = agg[(reader, fo, fn)]
            a1 = agg[(reader, fo, fn, "old_first")]
            a2 = agg[(reader, fo, fn, "new_first")]
            pct = lambda xs, k: 100 * sum(x[k] for x in xs) / len(xs)
            lines.append(f"{reader:<28}{fo + ' -> ' + fn:<26}{len(rs):>4}{pct(rs, 'correct'):>7.1f}%{pct(rs, 'stale'):>7.1f}%"
                         f"{pct(rs, 'abstained'):>6.1f}%{pct(a1, 'correct'):>12.1f}%{pct(a2, 'correct'):>12.1f}%")
        lines.append("")
    table = "\n".join(lines)
    (out / "summary.txt").write_text(table, encoding="utf-8")
    print(table)
    print("cloud usage:", ctx.client.cloud.usage if ctx.client.cloud else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
