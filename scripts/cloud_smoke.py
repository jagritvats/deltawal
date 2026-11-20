"""1-call cloud smoke test: guard blocks without --allow-cloud; with it, list models and make one call."""
from __future__ import annotations

import argparse
import json

from epochmem.cache import DiskCache
from epochmem.config import load_config, make_cloud_client, repo_path
from epochmem.guard import CloudCallBlocked, set_allow_cloud


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_dev_local.yaml")
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--provider", default="opencode-go")
    ap.add_argument("--allow-cloud", action="store_true", default=False)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cache = DiskCache(repo_path(cfg["runtime"]["cache_path"]))
    set_allow_cloud(False)
    try:
        make_cloud_client(cfg, cache, provider=args.provider)
        print("[FAIL] guard did not block cloud client without --allow-cloud")
        return 1
    except CloudCallBlocked:
        print("[PASS] guard blocks cloud endpoint without --allow-cloud")
    if not args.allow_cloud:
        return 0
    set_allow_cloud(True)
    client = make_cloud_client(cfg, cache, provider=args.provider)
    if args.list:
        print("models:", client.list_models())
    r = client.chat(args.model, [{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=64)
    print(json.dumps({"model": args.model, "content": r.content, "reasoning_chars": len(r.thinking),
                      "prompt_tokens": r.prompt_eval_count, "completion_tokens": r.eval_count,
                      "finish": r.done_reason, "wall_s": round(r.duration_s, 2), "cached": r.cached}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
