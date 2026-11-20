"""Write the held-out split (computed from the FULL dataset, dev excluded) to a small slice file so
run processes do not each parse the 277 MB source. Records are copied verbatim, in selection order."""
from __future__ import annotations

import argparse
import hashlib
import json

from epochmem.config import load_config, repo_path
from epochmem.runner import select_instances


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/m2_heldout.yaml")
    ap.add_argument("--out", default="data/raw/longmemeval_s_cleaned.heldout138.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["dataset"].pop("heldout_slice", None)
    qids = [p.inst.qid for p in select_instances(cfg, None)]
    src = repo_path(cfg["dataset"]["path"])
    with src.open("r", encoding="utf-8") as fh:
        raw = {r["question_id"]: r for r in json.load(fh)}
    records = [raw[q] for q in qids]
    out = repo_path(args.out)
    blob = json.dumps(records, ensure_ascii=False)
    out.write_text(blob, encoding="utf-8")
    meta = {"source": cfg["dataset"]["filename"], "source_sha256": cfg["dataset"]["sha256"], "n": len(records),
            "qids": qids, "slice_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {len(records)} records -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
