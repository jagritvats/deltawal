"""Zen (pay-as-you-go) spend ledger computed from the deterministic cache.

Every live cloud response is cached with its `usage` (incl. cached input tokens and the provider's billing
block). Spend = (uncached input x list price) + (cached input x cache-read price) + (output x output price),
skipping responses billed to the subscription (`usage.opencode.billing_source == "go"`). This formula matched
the console balance to within $0.02 on 2026-11-20. Use `--since` (unix time) to measure a stage/budget window.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time

from epochmem.config import repo_path

# USD per 1M tokens: (input, output, cached-input read). Zen docs, Nov 2026.
PRICES = {
    "glm-5.3-flash": (0.15, 0.50, 0.03), "glm-5.3": (1.40, 4.40, 0.26), "gpt-6-luna": (0.10, 0.50, 0.01),
    "gpt-5.6-luna": (0.20, 1.20, 0.02), "qwen3.8-flash": (0.15, 0.47, 0.016), "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


def spend_since(since: float, cache_path: str | None = None) -> dict:
    c = sqlite3.connect(str(repo_path(cache_path or ".cache/llm_cache.sqlite")))
    per = {}
    for m, (pin, pout, pc) in PRICES.items():
        cost = calls = 0
        for (s,) in c.execute("select value from cache where kind='chat' and model=? and created>?", (m, since)):
            u = json.loads(s).get("usage") or {}
            if (u.get("opencode") or {}).get("billing_source") == "go":
                continue
            pt, ct = u.get("prompt_tokens", 0) or 0, u.get("completion_tokens", 0) or 0
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
            cost += (pt - cached) / 1e6 * pin + cached / 1e6 * pc + ct / 1e6 * pout
            calls += 1
        if calls:
            per[m] = {"calls": calls, "usd": round(cost, 4)}
    return {"since": since, "total_usd": round(sum(v["usd"] for v in per.values()), 4), "by_model": per}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, default=None, help="unix time; default = runs/budget_start.json")
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--init", action="store_true", help="write runs/budget_start.json with now() and --cap")
    args = ap.parse_args()
    bpath = repo_path("runs/budget_start.json")
    if args.init:
        bpath.write_text(json.dumps({"since": time.time(), "cap_usd": args.cap}), encoding="utf-8")
    b = json.loads(bpath.read_text(encoding="utf-8")) if bpath.exists() else {"since": 0, "cap_usd": None}
    since = args.since if args.since is not None else b["since"]
    r = spend_since(since)
    cap = args.cap if args.cap is not None else b.get("cap_usd")
    if cap is not None:
        r["cap_usd"], r["remaining_under_cap_usd"] = cap, round(cap - r["total_usd"], 4)
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
