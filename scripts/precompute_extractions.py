"""Fill the extraction cache for every (question, commit order) the grid will need, in parallel.

With memory.registry_mode=commit_order, cells with p_ooo>0 extract sessions in worker commit
order using a committed-only key registry; p_ooo=0 (and B1) use the chronological pass.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from epochmem.config import load_config
from epochmem.guard import set_allow_cloud
from epochmem.runner import build_ctx, select_instances
from epochmem.stream.schedule import lagged_commit_order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_dev_local.yaml")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    ap.add_argument("--allow-go-scale", action="store_true", default=False)
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    cfg.setdefault("runtime", {})["allow_go_scale"] = args.allow_go_scale
    from epochmem.runner import enforce_provider_caps
    ctx, _ = build_ctx(cfg, session_id="ses_epochmem_extract")
    preps = select_instances(cfg, None)
    enforce_provider_caps(cfg, len(preps))
    jobs, later = [], []
    for prep in preps:
        jobs.append((prep, None))
        if cfg["memory"].get("registry_mode") == "commit_order":
            for p in cfg["grid"]["p_ooo"]:
                if p > 0:
                    for lag in cfg["grid"].get("p_ooo_lags", cfg["grid"]["lags"]):
                        later.append((lag, prep, lagged_commit_order(prep.n, lag, float(p),
                                                                     f"{cfg['seed']}:{prep.inst.qid}:{float(p)}")))

    def run(job):
        prep, order = job
        ctx.extractor.extract_instance(prep, order)
        print(f"{prep.inst.qid} {'chrono' if order is None else 'commit_order'} done", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:      # all chronological passes first, then commit orders
        list(ex.map(run, jobs))
        # committed-only chains: one lag at a time so a question's chains never race on a shared prefix
        for lag in sorted({l for l, _, _ in later}):
            list(ex.map(run, [(prep, order) for l, prep, order in later if l == lag]))
    print("stats", ctx.extractor.stats)
    usage = {m: u for c in ctx.client.clouds.values() for m, u in c.usage_by_model.items()}
    spent = {p: round(c.spent_usd(), 4) for p, c in ctx.client.clouds.items()}
    print("usage", json.dumps(usage), "rate_limited", sum(c.stats.rate_limited for c in ctx.client.clouds.values()),
          "spent_usd", json.dumps(spent))
    from epochmem.config import repo_path
    (repo_path("runs") / "precompute_heldout_usage.json").write_text(json.dumps(usage), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
