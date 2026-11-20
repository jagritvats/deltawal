"""Fill the extraction cache for every (question, commit order) the grid will need, in parallel.

With memory.registry_mode=commit_order, cells with p_ooo>0 extract sessions in worker commit
order using a committed-only key registry; p_ooo=0 (and B1) use the chronological pass.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

from epochmem.config import load_config
from epochmem.guard import set_allow_cloud
from epochmem.runner import build_ctx, select_instances
from epochmem.stream.schedule import commit_order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_dev_local.yaml")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    set_allow_cloud(False)
    cfg = load_config(args.config)
    ctx, _ = build_ctx(cfg)
    jobs = []
    for prep in select_instances(cfg, None):
        jobs.append((prep, None))
        if cfg["memory"].get("registry_mode") == "commit_order":
            for p in cfg["grid"]["p_ooo"]:
                if p > 0:
                    jobs.append((prep, commit_order(prep.n, float(p), f"{cfg['seed']}:{prep.inst.qid}:{float(p)}")))

    def run(job):
        prep, order = job
        ctx.extractor.extract_instance(prep, order)
        print(f"{prep.inst.qid} {'chrono' if order is None else 'commit_order'} done", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(run, jobs))
    print("stats", ctx.extractor.stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
