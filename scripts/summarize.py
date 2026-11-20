"""Re-print the summary table for a finished (or partial) run."""
from __future__ import annotations

import argparse
import json
import sys

from epochmem.config import repo_path
from epochmem.eval.metrics import aggregate, format_table, load_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    agg = aggregate(load_rows(repo_path("runs") / args.run_id / "results.jsonl"))
    print(json.dumps(agg, indent=2) if args.json else format_table(agg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
