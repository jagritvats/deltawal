"""Download LongMemEval-S (cleaned) at a pinned revision and verify sha256 + size.

Also warms the o200k tokenizer used for prompt-token counting.
"""
from __future__ import annotations

import argparse
import sys

import httpx

from epochmem.config import load_config, repo_path
from epochmem.data.longmemeval import sha256_file


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m1_local.yaml")
    args = ap.parse_args()
    d = load_config(args.config)["dataset"]
    if "oracle" in d["filename"]:
        raise SystemExit("refusing to download the oracle split")
    dest = repo_path(d["path"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha256_file(dest) == d["sha256"]:
        print(f"already present and verified: {dest}")
    else:
        url = f"https://huggingface.co/datasets/{d['hf_repo']}/resolve/{d['hf_revision']}/{d['filename']}"
        tmp = dest.with_suffix(".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in r.iter_bytes(1 << 20):
                    fh.write(chunk)
        digest = sha256_file(tmp)
        if digest != d["sha256"] or tmp.stat().st_size != d["size_bytes"]:
            tmp.unlink()
            raise SystemExit(f"verification failed: sha256={digest}")
        tmp.replace(dest)
        print(f"downloaded and verified: {dest}")
    from epochmem.tokens import count_text
    count_text("warm")
    print("tokenizer ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
