"""Cloud-only twin of scripts/precompute_extractions.py for the chronological pass.

`runner.build_ctx` also opens Kafka (ensure_topic) and Postgres (DDL on connect); this script builds ONLY
the model router + FactExtractor with the exact same parameters as build_ctx, so every request (and hence
every SQLite cache key) is byte-identical to what run_m1.py will later issue. No Ollama call, no Kafka,
no Postgres, no Redis. Commit-order passes (p_ooo > 0) are refused: use the original script for those.

Also writes a per-question evidence-coverage report (facts extracted in every evidence session).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from epochmem.cache import DiskCache
from epochmem.config import load_config, make_cloud_client, repo_path
from epochmem.guard import set_allow_cloud
from epochmem.llm import ModelRouter, OllamaClient, provider_of
from epochmem.memory.extract import FactExtractor
from epochmem.runner import enforce_provider_caps, select_instances


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    ap.add_argument("--out", default=None, help="coverage/usage JSON (default runs/precompute_<slice stem>.json)")
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    if any(float(p) > 0 for p in cfg["grid"]["p_ooo"]):
        raise SystemExit("p_ooo > 0 in grid: commit-order passes need scripts/precompute_extractions.py")
    m, mem = cfg["models"], cfg["memory"]
    if provider_of(str(m["extractor"])) is None:
        raise SystemExit("extractor is a local model; this script is cloud-only")

    # --- identical to runner.build_ctx (client + extractor only) ---
    cache = DiskCache(repo_path(cfg["runtime"]["cache_path"]))
    local = OllamaClient(m["endpoint"], cache, num_ctx=m["num_ctx"], temperature=m["temperature"],
                         seed=cfg["seed"], think=m["think"])            # constructed, never called
    prov = provider_of(str(m["extractor"]))
    clouds = {prov: make_cloud_client(cfg, cache, session_id="ses_epochmem_extract", provider=prov)}
    client = ModelRouter(local, clouds, min_cloud_max_tokens=(cfg.get("cloud") or {}).get("min_max_tokens", 2048))
    extractor = FactExtractor(client, m["extractor"], max_facts=mem["max_facts_per_session"],
                              registry_max_keys=mem["registry_max_keys"], asst_chars=mem["assistant_trunc_chars_extract"])

    preps = select_instances(cfg, None)
    enforce_provider_caps(cfg, len(preps))
    t0 = time.time()
    print(f"start_unix {t0:.3f}; {len(preps)} instances, {sum(p.n for p in preps)} sessions", flush=True)

    def run(prep):
        facts = extractor.extract_instance(prep, None)
        ev = prep.evidence_epochs()
        counts = [len(facts.get(e, [])) for e in ev]
        print(f"{prep.inst.qid} chrono done ev_facts={counts}", flush=True)
        return {"qid": prep.inst.qid, "qtype": prep.inst.qtype, "n_sessions": prep.n,
                "n_facts": sum(len(v) for v in facts.values()), "evidence_epochs": ev,
                "evidence_fact_counts": counts, "all_evidence_extracted": bool(ev) and all(counts),
                "empty_sessions": sum(1 for v in facts.values() if not v)}

    with ThreadPoolExecutor(args.workers) as ex:
        per_q = list(ex.map(run, preps))

    by_type: dict[str, dict] = {}
    for r in per_q:
        b = by_type.setdefault(r["qtype"], {"n": 0, "all_evidence_extracted": 0, "ev_sessions": 0, "ev_sessions_with_facts": 0})
        b["n"] += 1
        b["all_evidence_extracted"] += int(r["all_evidence_extracted"])
        b["ev_sessions"] += len(r["evidence_fact_counts"])
        b["ev_sessions_with_facts"] += sum(1 for c in r["evidence_fact_counts"] if c)
    st = extractor.stats
    usage = {mm: u for c in clouds.values() for mm, u in c.usage_by_model.items()}
    report = {
        "config": args.config, "start_unix": t0, "wall_s": round(time.time() - t0, 1),
        "extraction": {"calls": st.calls, "parse_failures": st.parse_failures, "retries": st.retries,
                       "cached": st.cached, "facts": st.facts},
        "live_cloud_calls": sum(c.stats.chat_calls for c in clouds.values()),
        "usage": usage, "rate_limited": sum(c.stats.rate_limited for c in clouds.values()),
        "client_est_spent_usd": {p: round(c.spent_usd(), 4) for p, c in clouds.items()},
        "coverage": {"n": len(per_q), "all_evidence_extracted": sum(r["all_evidence_extracted"] for r in per_q),
                     "by_type": by_type},
        "per_question": per_q,
    }
    out = repo_path(args.out) if args.out else repo_path("runs") / (
        "precompute_" + repo_path(cfg["dataset"]["heldout_slice"]).stem.split(".")[-1] + ".json")
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("stats", json.dumps(report["extraction"]), "live_calls", report["live_cloud_calls"],
          "usage", json.dumps(usage), "rate_limited", report["rate_limited"],
          "client_est_spent_usd", json.dumps(report["client_est_spent_usd"]))
    print("coverage", json.dumps(report["coverage"]))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
