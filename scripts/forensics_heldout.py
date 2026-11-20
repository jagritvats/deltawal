"""Forensic diagnostics on held-out results. ZERO network calls: reads run files; any extraction
lookups are cache-only (cloud clients patched to raise).

  #1 deltawal_prompt (lag=1,p=0) RIGHT but B1_sync (lag=0) WRONG on KU: structural categorization.
  #4 deterministic (lag=0, p_ooo=0.3) LWW clobber simulation over cached committed-only facts.
  #5 keys with >=2 committed versions before the lag=1 cutoff.
  #2 gemma temporal: B1/lazy RIGHT but deltawal_prompt WRONG: show the failure pattern.
  #3 p_ooo=0.3 QA impact, OOCR and missed-merge, and whether corrupted keys were surfaced/queried.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from epochmem.config import load_config, repo_path
from epochmem.guard import set_allow_cloud
from epochmem.runner import build_ctx, select_instances
from epochmem.stream.schedule import lagged_commit_order

STOP = set("the a an of to in on my i me you your is was were be been for and or at with that this it as by from "
           "what how many much did do does have has had time times there are which when where who whom".split())


def toks(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$.,:]+", (s or "").lower().replace(",", "")) if w not in STOP and len(w) > 1}


def load(run):
    rows = [json.loads(l) for l in open(repo_path("runs") / run / "results_judged.jsonl", encoding="utf-8")]
    return {(r["system"], r["lag"], r["p_ooo"], r["qid"]): r for r in rows}


def main() -> int:
    set_allow_cloud(True)
    cfg = load_config("configs/m2_heldout.yaml")
    preps = {p.inst.qid: p for p in select_instances(cfg, None)}
    ctx, _ = build_ctx(cfg, session_id="ses_forensics")
    for c in ctx.client.clouds.values():                    # cache-only: any miss raises
        c.http.post = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cache-only"))
    out = {}

    # ---------------------------------------------------------------- #1
    print("=" * 100 + "\n#1  KU: deltawal_prompt(lag=1,p=0) RIGHT  vs  B1_sync(lag=0) WRONG  (structural categorization)\n" + "=" * 100)
    diag = {json.loads(l)["qid"]: json.loads(l) for l in open(repo_path("runs/m2ho_gemma/extraction_diag.jsonl"), encoding="utf-8")}
    past_re = re.compile(r"\b(initial|initially|originally|first|before|previous|previously|used to|at first|earlier|prior|"
                         r"than i did|than before|switch|switched)\b", re.I)
    cats_all = Counter()
    for run in ("m2ho_glm", "m2ho_luna6"):
        R = load(run)
        cats = Counter()
        for qid, p in preps.items():
            if p.inst.qtype != "knowledge-update" or p.inst.is_abstention:
                continue
            b1, pr = R.get(("B1_sync", 0, 0.0, qid)), R.get(("deltawal_prompt", 1, 0.0, qid))
            if not (b1 and pr and pr["correct"] and not b1["correct"]):
                continue
            ev = p.evidence_epochs()
            old_ev, new_ev = set(ev[:-1]), ev[-1]
            fbe = {int(k): v for k, v in diag[qid]["facts_by_evidence_epoch"].items()}
            new_f = fbe.get(new_ev, [])
            old_f = [f for e in old_ev for f in fbe.get(e, [])]
            collide = {f["entity_key"] for f in new_f} & {f["entity_key"] for f in old_f}
            qt = toks(p.inst.question)
            # question-relevant overwritten key: collided key whose name/new fact best overlaps the question
            rel = max(collide, key=lambda k: len(qt & toks(k.replace(".", " ").replace("_", " ") + " " + " ".join(
                f["fact_text"] for f in new_f if f["entity_key"] == k))), default=None)
            past = bool(past_re.search(p.inst.question))
            ret = b1["retrieved"]
            if past and rel:
                # B1 commits in order with LWW, so a collided key holds ONLY the newer value by construction
                cat = "(a) destructive overwrite: past/comparative Q on a key the update overwrote"
            elif not collide and any(f["source_epoch"] in old_ev for f in ret) and any(f["source_epoch"] == new_ev for f in ret):
                cat = "(b) unordered missed merge: old+new under different keys"
            elif not b1["answer"].strip():
                cat = "(f) empty reader output"
            else:
                g = toks(p.inst.answer)
                new_raw = " ".join(t.content for s_ in p.stream if s_.epoch == new_ev for t in s_.session.turns if t.has_answer)
                in_facts = len(g & toks(" ".join(f["fact_text"] for f in new_f))) >= max(1, len(g) // 2)
                in_raw = len(g & toks(new_raw)) >= max(1, len(g) // 2)
                cat = "(c) lossy extraction" if (not in_facts and in_raw) else "(e) reader hedged/abstained with current value present"
            cats[cat] += 1
            print(f"  {run:11s} {qid:15s} {cat[:4]} past={past!s:5s} overwritten_key={rel} | Q: {p.inst.question[:70]}")
        print(f"--- {run}: {sum(cats.values())} cases")
        for c, n in cats.most_common():
            print(f"    {n:3d}  {c}")
        cats_all.update(cats)
        out[f"q1_{run}"] = dict(cats)
    print("--- combined:", dict(cats_all))
    out["q1_combined"] = dict(cats_all)

    # ---------------------------------------------------------------- #2
    print("\n" + "=" * 100 + "\n#2  gemma TEMPORAL: B1(lag0) or lazy(lag1) RIGHT, deltawal_prompt(lag1,p0) WRONG\n" + "=" * 100)
    import inspect
    from epochmem.systems import deltawal
    src = inspect.getsource(deltawal._DeltaWALBase._turn_line) + inspect.getsource(deltawal._DeltaWALBase._fact_line)
    print("WAL/fact line templates:\n" + "\n".join("   " + l.strip() for l in src.splitlines() if "return" in l))
    R = load("m2ho_gemma")
    fails = []
    for qid, p in preps.items():
        if p.inst.qtype != "temporal-reasoning" or p.inst.is_abstention:
            continue
        b1, lz, pr = (R.get(k) for k in (("B1_sync", 0, 0.0, qid), ("deltawal_lazy", 1, 0.0, qid), ("deltawal_prompt", 1, 0.0, qid)))
        if pr and not pr["correct"] and ((b1 and b1["correct"]) or (lz and lz["correct"])):
            fails.append((qid, p, b1, lz, pr))
    print(f"\n{len(fails)} such questions; first 10:")
    pat = Counter()
    for qid, p, b1, lz, pr in fails[:10]:
        gated = pr["read_diag"].get("gated", [])
        print(f"\n  {qid}: Q: {p.inst.question}\n    gold: {p.inst.answer[:90]}\n    question_date: {p.inst.question_date}"
              f"\n    prompt ans: {pr['answer'][:160]}\n    B1 ans    : {(b1 or {}).get('answer','')[:160]}"
              f"\n    lazy ans  : {(lz or {}).get('answer','')[:160]}"
              f"\n    prompt ctx: {len(pr['retrieved'])} facts + {len(gated)} gated WAL turns (epochs {sorted({g[0] for g in gated})}); "
              f"abstained={pr['abstained']} tokens={pr['prompt_tokens']} vs B1 {(b1 or {}).get('prompt_tokens')}")
        a = pr["answer"].lower()
        pat["abstained / says unknown"] += bool(pr["abstained"])
        pat["mentions 'epoch'"] += "epoch" in a
        pat["answers with a different count/date (committed)"] += (not pr["abstained"])
    print("\n  pattern counts (10 shown):", dict(pat))
    out["q2_n"] = len(fails)

    # ---------------------------------------------------------------- #3
    print("\n" + "=" * 100 + "\n#3  p_ooo=0.3 impact: QA accuracy, OOCR, missed-merge, corrupted-key exposure\n" + "=" * 100)
    for run in ("m2ho_gemma", "m2ho_glm", "m2ho_luna6"):
        rows = [json.loads(l) for l in open(repo_path("runs") / run / "results_judged.jsonl", encoding="utf-8")]
        cell = defaultdict(list)
        for r in rows:
            cell[(r["system"], r["lag"], r["p_ooo"])].append(r)
        print(f"\n--- {run} ({rows[0]['reader_model']})")
        print(f"    {'system':<24}{'lag':>4}{'p':>6} | {'Acc':>6}{'KU':>7}{'TR':>7}{'Abs':>7} | {'OOCR':>6} {'#upd':>5} {'MissM':>6} | "
              f"{'rows w/ corrupt':>15} {'corrupt key shown':>18} {'corrupt evid. key':>18}")
        for (s, lag, p) in sorted(cell, key=lambda k: (k[0], k[1], k[2])):
            if s not in ("B2_async", "B3_recency_matched_user", "deltawal_prompt", "deltawal_shadow_v2", "deltawal_lazy"):
                continue
            rs = cell[(s, lag, p)]
            ku = [r for r in rs if r["qtype"] == "knowledge-update" and not r["is_abstention"]]
            tr = [r for r in rs if r["qtype"] == "temporal-reasoning" and not r["is_abstention"]]
            ab = [r for r in rs if r["is_abstention"]]
            up = sum(r["oocr_updated"] for r in rs)
            co = sum(r["oocr_corrupted"] for r in rs)
            mm = [r["missed_merge"] for r in ku if r.get("missed_merge") is not None]
            wc = [r for r in rs if r["corrupted_keys"]]
            shown = sum(1 for r in wc if {f["entity_key"] for f in r["retrieved"]} & set(r["corrupted_keys"]))
            # evidence-key corruption: recompute this cell's commit-order facts (cache-only) and test overlap
            ev_hit = 0
            if p > 0 and wc:
                for r in wc:
                    pr = preps[r["qid"]]
                    order = lagged_commit_order(pr.n, lag, p, f"{cfg['seed']}:{pr.inst.qid}:{p}")
                    fbe = ctx.extractor.extract_instance(pr, order)
                    evk = {f.entity_key for e in pr.evidence_epochs() for f in fbe.get(e, [])}
                    ev_hit += bool(evk & set(r["corrupted_keys"]))
            pct = lambda xs: f"{100 * sum(x['correct'] for x in xs) / len(xs):5.1f}" if xs else "    -"
            print(f"    {s:<24}{lag:>4}{p:>6.2f} | {pct(rs):>6}{pct(ku):>7}{pct(tr):>7}{pct(ab):>7} | "
                  f"{(100 * co / up if up else 0):6.1f} {up:5d} {(100 * sum(mm) / len(mm) if mm else 0):6.1f} | "
                  f"{len(wc):15d} {shown:18d} {ev_hit:18d}")
    # ---------------------------------------------------------------- #4
    print("\n" + "=" * 100 + "\n#4  Deterministic (lag=0, p_ooo=0.3) LWW clobber simulation  (ZERO LLM calls)\n" + "=" * 100)
    tot_up = tot_cor = tot_cor_cas = 0
    q_ev_cor = q_any_cor = 0
    ku_final_clobbered, ku_rel_clobbered, approx_sessions, ku_n = [], [], 0, 0
    for qid, pr in preps.items():
        key = f"{cfg['seed']}:{qid}:0.3"
        order0 = lagged_commit_order(pr.n, 0, 0.3, key)
        order1 = lagged_commit_order(pr.n, 1, 0.3, key)
        fbe = ctx.extractor.extract_instance(pr, order1)   # cached committed-only facts (approximation, see count)
        approx_sessions += len(order0) - 1 - order0.index(pr.n - 1)    # sessions committed after the final one
        lww, cas, writers = {}, {}, defaultdict(set)
        for idx in order0:
            e = idx + 1
            for f in fbe.get(e, []):
                lww[f.entity_key] = e
                cas[f.entity_key] = max(cas.get(f.entity_key, 0), e)
                writers[f.entity_key].add(e)
        upd = [k for k, es in writers.items() if len(es) >= 2]
        cor = {k for k in upd if lww[k] < max(writers[k])}
        tot_up += len(upd)
        tot_cor += len(cor)
        tot_cor_cas += sum(1 for k in upd if cas[k] < max(writers[k]))
        evk = {f.entity_key for e in pr.evidence_epochs() for f in fbe.get(e, [])}
        q_any_cor += bool(cor)
        q_ev_cor += bool(cor & evk)
        if pr.inst.qtype == "knowledge-update" and not pr.inst.is_abstention:
            ku_n += 1
            L = pr.evidence_epochs()[-1]
            lkeys = {f.entity_key for f in fbe.get(L, [])}
            if any(lww.get(k, L) < L for k in lkeys):
                ku_final_clobbered.append(qid)
            old_keys = {f.entity_key for e in pr.evidence_epochs()[:-1] for f in fbe.get(e, [])}
            if any(lww.get(k, L) < L for k in lkeys & old_keys):
                ku_rel_clobbered.append(qid)
    r4 = {"oocr_lww": round(tot_cor / tot_up, 4) if tot_up else None, "updated_keys": tot_up, "corrupted_keys": tot_cor,
          "oocr_cas": round(tot_cor_cas / tot_up, 4) if tot_up else None,
          "questions_with_any_corrupted_key": q_any_cor, "questions_with_corrupted_evidence_key": q_ev_cor,
          "ku_questions": ku_n,
          "ku_final_update_any_key_overwritten_by_older": len(ku_final_clobbered),
          "ku_shared_old_new_key_overwritten_by_older": len(ku_rel_clobbered),
          "ku_clobbered_qids": ku_rel_clobbered,
          "approximation_sessions_committed_after_final_session": approx_sessions}
    print(json.dumps(r4, indent=1))
    out["q4_lag0_p03"] = r4

    # ---------------------------------------------------------------- #5
    print("\n" + "=" * 100 + "\n#5  Keys with >=2 COMMITTED versions before the lag=1 cutoff\n" + "=" * 100)
    q_any_multi = q_ev_multi = q_ku_ev_multi = 0
    for qid, pr in preps.items():
        fbe = ctx.extractor.extract_instance(pr)           # chronological pass (p_ooo=0 cells)
        vers = defaultdict(set)
        for e in range(1, pr.n):                           # lag=1: final session still in flight
            for f in fbe.get(e, []):
                vers[f.entity_key].add(e)
        multi = {k for k, es in vers.items() if len(es) >= 2}
        evk = {f.entity_key for e in pr.evidence_epochs() if e < pr.n for f in fbe.get(e, [])}
        q_any_multi += bool(multi)
        q_ev_multi += bool(multi & evk)
        if pr.inst.qtype == "knowledge-update" and not pr.inst.is_abstention:
            q_ku_ev_multi += bool(multi & evk)
    r5 = {"questions_with_any_key_having_2plus_committed_versions": q_any_multi,
          "questions_where_an_evidence_key_has_2plus_committed_versions": q_ev_multi,
          "ku_questions_where_an_evidence_key_has_2plus_committed_versions": q_ku_ev_multi,
          "n_questions": len(preps)}
    print(json.dumps(r5, indent=1))
    out["q5_versions_before_lag1_cutoff"] = r5

    (repo_path("runs") / "forensics_heldout.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    calls = sum(c.stats.chat_calls for c in ctx.client.clouds.values())
    print(f"\nnetwork chat calls made: {calls} (must be 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
