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
    ap.add_argument("--allow-cloud", action="store_true", default=False,
                    help="permit non-localhost model endpoints (off for Milestone 1)")
    args = ap.parse_args()
    set_allow_cloud(args.allow_cloud)
    cfg = load_config(args.config)
    systems = args.systems or cfg["grid"]["systems"]
    run_dir = run(cfg, args.run_id, systems, n_smoke=args.smoke)
    table = format_table(aggregate(load_rows(run_dir / "results.jsonl")))
    (run_dir / "summary.txt").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
