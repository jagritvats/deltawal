"""Milestone 1 local-only dry run. Resumable: re-running with the same --run-id skips finished rows."""
from __future__ import annotations

import argparse
import sys

from epochmem.config import load_config
from epochmem.eval.metrics import aggregate, format_table, load_rows
from epochmem.guard import set_allow_cloud
from epochmem.runner import run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_local.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", type=int, default=None, help="only run N instances (round-robin over strata)")
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--reader", default=None, help="override reader model (e.g. opencode-go/glm-5.3)")
    ap.add_argument("--judge", default=None, help="override judge model (e.g. opencode-go/qwen4-max)")
    ap.add_argument("--core", action="store_true", help="lean core cells only (B1 lag0; async lag1/3 p0, lag1 p0.3)")
    ap.add_argument("--cells", default=None, help="restrict core async cells, e.g. '1:0,3:0' (lag:p_ooo)")
    ap.add_argument("--workers", type=int, default=None, help="instances processed concurrently")
    ap.add_argument("--allow-cloud", action="store_true", default=False,
                    help="permit non-localhost model endpoints (off for Milestone 1)")
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    if args.reader:
        cfg["models"]["reader"] = args.reader
    if args.judge:
        cfg["models"]["judge"] = args.judge
    if args.workers:
        cfg["runtime"]["workers"] = args.workers
    if args.cells:
        import epochmem.runner as R
        R.CORE_CELLS = [(int(a), float(b)) for a, b in (c.split(":") for c in args.cells.split(","))]
    systems = args.systems or cfg["grid"]["systems"]
    run_dir = run(cfg, args.run_id, systems, n_smoke=args.smoke, core=args.core)
    table = format_table(aggregate(load_rows(run_dir / "results.jsonl")))
    (run_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
