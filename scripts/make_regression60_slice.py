"""Stage B regression slice: 60 non-abstention LongMemEval-S (cleaned) questions of the types the M2 splits
never touched (30 multi-session, 10 single-session-user, 10 single-session-assistant,
10 single-session-preference), with ZERO overlap with the 20 dev and 138 held-out questions.

Sampling: per stratum, `random.Random(f"{seed}:regression60:{stratum}").sample(pool, n)` over the
qid-sorted pool (strata in sorted order). Records are copied verbatim, in selection order, exactly like
scripts/make_heldout_slice.py. The 277 MB source is read once; dev/held-out membership is recomputed from
that same in-memory copy via `epochmem.runner.select_instances`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random

import epochmem.data.longmemeval as LM
from epochmem.config import load_config, repo_path
from epochmem.runner import select_instances

SPEC = {"multi-session": 30, "single-session-user": 10, "single-session-assistant": 10,
        "single-session-preference": 10}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-config", default="configs/m2_dev_local.yaml")
    ap.add_argument("--heldout-config", default="configs/m2_heldout.yaml")
    ap.add_argument("--heldout-meta", default="data/raw/longmemeval_s_cleaned.heldout138.meta.json")
    ap.add_argument("--out", default="data/raw/longmemeval_s_cleaned.regression60.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dev_cfg = load_config(args.dev_config)
    src = repo_path(dev_cfg["dataset"]["path"])
    blob_in = src.read_bytes()                                   # single read of the 277 MB source
    digest = hashlib.sha256(blob_in).hexdigest()
    assert digest == dev_cfg["dataset"]["sha256"], f"source sha256 mismatch: {digest}"
    raw_list = json.loads(blob_in.decode("utf-8"))
    del blob_in
    raw = {r["question_id"]: r for r in raw_list}

    # Serve the in-memory copy to the repo loader so select_instances() does not re-parse the file.
    class _InMemJSON:
        @staticmethod
        def load(_fh):
            return raw_list
    real_json = LM.json
    LM.json = _InMemJSON
    try:
        dev_qids = [p.inst.qid for p in select_instances(dev_cfg, None)]
        ho_cfg = load_config(args.heldout_config)
        ho_cfg["dataset"].pop("heldout_slice", None)
        ho_recomputed = [p.inst.qid for p in select_instances(ho_cfg, None)]
        insts = LM.LongMemEvalDataset(src).load()
    finally:
        LM.json = real_json

    ho_meta = json.loads(repo_path(args.heldout_meta).read_text(encoding="utf-8"))
    ho_qids = list(ho_meta["qids"])
    assert len(dev_qids) == 20 and len(set(dev_qids)) == 20, len(dev_qids)
    assert len(ho_qids) == 138 and len(set(ho_qids)) == 138, len(ho_qids)
    assert ho_recomputed == ho_qids, "held-out meta does not match a recomputation from the source"
    excluded = set(dev_qids) | set(ho_qids)
    assert len(excluded) == 158, len(excluded)

    picked: list[str] = []
    pools = {}
    for stratum in sorted(SPEC):
        pool = sorted((i for i in insts if i.stratum == stratum and not i.is_abstention
                       and not i.qid.endswith("_abs") and i.qid not in excluded), key=lambda i: i.qid)
        pools[stratum] = len(pool)
        n = SPEC[stratum]
        if len(pool) < n:
            raise ValueError(f"stratum {stratum!r}: need {n}, have {len(pool)}")
        picked.extend(i.qid for i in random.Random(f"{args.seed}:regression60:{stratum}").sample(pool, n))

    # Hard guarantees: 60 unique, non-abstention, right composition, zero overlap with dev + held-out.
    overlap = set(picked) & excluded
    assert not overlap, f"overlap with dev/held-out: {sorted(overlap)}"
    assert not (set(picked) & set(dev_qids)) and not (set(picked) & set(ho_qids))
    assert len(picked) == len(set(picked)) == sum(SPEC.values())
    assert not any(q.endswith("_abs") for q in picked)
    comp: dict[str, int] = {}
    for q in picked:
        comp[raw[q]["question_type"]] = comp.get(raw[q]["question_type"], 0) + 1
    assert comp == SPEC, comp

    records = [raw[q] for q in picked]
    out = repo_path(args.out)
    assert "s_cleaned" in out.name
    blob = json.dumps(records, ensure_ascii=False)
    out.write_text(blob, encoding="utf-8")
    meta = {"source": dev_cfg["dataset"]["filename"], "source_sha256": digest, "n": len(records),
            "qids": picked, "slice_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
            "seed": args.seed, "rng_key": f"{args.seed}:regression60:<stratum>", "spec": SPEC,
            "pool_sizes_after_exclusion": pools, "excluded_n": len(excluded), "overlap_with_dev_or_heldout": 0}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"composition {comp}; pools after exclusion {pools}; overlap with {len(excluded)} used qids: 0")
    print(f"wrote {len(records)} records -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
