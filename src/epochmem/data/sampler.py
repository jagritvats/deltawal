"""Deterministic stratified sampling."""
from __future__ import annotations

import random

from .base import Instance


def stratified_sample(instances: list[Instance], spec: dict[str, int], seed: int) -> list[Instance]:
    """spec maps stratum -> count. Stratum 'abstention' = any `_abs` question; other strata
    are question types restricted to non-abstention questions."""
    out: list[Instance] = []
    for stratum in sorted(spec):
        pool = sorted((i for i in instances if i.stratum == stratum), key=lambda i: i.qid)
        n = spec[stratum]
        if len(pool) < n:
            raise ValueError(f"stratum {stratum!r}: need {n}, have {len(pool)}")
        rng = random.Random(f"{seed}:{stratum}")
        out.extend(rng.sample(pool, n))
    return out


def heldout_sample(instances: list[Instance], spec: dict, dev_spec: dict, seed: int) -> list[Instance]:
    """Held-out split: per stratum either 'all' or a count, always EXCLUDING the dev sample drawn
    with `dev_spec` (same seed), so dev and held-out never overlap."""
    dev_ids = {i.qid for i in stratified_sample(instances, dev_spec, seed)}
    out: list[Instance] = []
    for stratum in sorted(spec):
        pool = sorted((i for i in instances if i.stratum == stratum and i.qid not in dev_ids), key=lambda i: i.qid)
        n = spec[stratum]
        if n == "all":
            out.extend(pool)
            continue
        if len(pool) < n:
            raise ValueError(f"held-out stratum {stratum!r}: need {n}, have {len(pool)}")
        out.extend(random.Random(f"{seed}:heldout:{stratum}").sample(pool, n))
    assert not ({i.qid for i in out} & dev_ids)
    return out


def smoke_subset(sample: list[Instance], n: int) -> list[Instance]:
    """Pick n instances round-robin across strata, preserving sample order."""
    by: dict[str, list[Instance]] = {}
    for inst in sample:
        by.setdefault(inst.stratum, []).append(inst)
    out: list[Instance] = []
    keys = sorted(by)
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop(0))
    return out
