"""WAL snapshot (single Lua round trip) must equal committed() + delta(); trim must drop committed turns."""
from __future__ import annotations

import uuid

import numpy as np
import pytest

from epochmem.config import load_config
from epochmem.data.base import Session, Turn
from epochmem.data.stream import StreamSession

redis = pytest.importorskip("redis")


@pytest.fixture()
def wal():
    from epochmem.store.wal import RedisWAL
    w = RedisWAL(load_config("configs/m1_local.yaml")["infra"]["redis_url"], trim=True)
    try:
        w.ping()
    except redis.exceptions.ConnectionError as e:
        pytest.skip(f"redis unavailable: {e}")
    return w


def _ss(epoch):
    return StreamSession(epoch, Session(f"s{epoch}", "d", (Turn("user", f"u{epoch}"), Turn("assistant", f"a{epoch}"))), False)


def test_snapshot_matches_delta_and_trim(wal):
    ns = f"t-{uuid.uuid4()}"
    for e in range(1, 7):
        assert wal.next_epoch(ns) == e
        wal.append_session(ns, e, _ss(e), [[float(e), float(t)] for t in range(2)])
    for e in (1, 2, 4):                      # 3 is a straggler: watermark stops at 2
        wal.mark_applied(ns, e)
    assert wal.committed(ns) == 2
    assert wal.log_size(ns) == 8             # epochs 1-2 trimmed (2 turns each), 3-6 kept
    wm, snap = wal.snapshot(ns)
    ref = wal.delta(ns, wal.committed(ns), uncommitted_only=True, with_embeddings=True)
    assert wm == 2
    assert [(r["epoch"], r["turn"]) for r in snap] == [(r["epoch"], r["turn"]) for r in ref] == \
           [(3, 0), (3, 1), (5, 0), (5, 1), (6, 0), (6, 1)]          # 4 is applied -> excluded
    assert all(np.array_equal(a["embedding"], b["embedding"]) for a, b in zip(snap, ref))
    wal.mark_applied(ns, 3)
    assert wal.committed(ns) == 4 and wal.log_size(ns) == 4
    wal.reset(ns)


def test_async_ack_pending_then_lazy_fill(wal):
    from epochmem.memory.wal_embed import embed_records
    ns = f"t-{uuid.uuid4()}"
    for e in (1, 2):
        wal.next_epoch(ns)
        wal.append_session(ns, e, _ss(e))                 # ack path: no embeddings
    assert wal.pop_pending(ns) == 1 and wal.pop_pending(ns) == 2 and wal.pop_pending(ns) is None
    wm, snap = wal.snapshot(ns)
    assert wm == 0 and all(r["embedding"] is None and r["sent_embeddings"] is None for r in snap)
    w2 = type(wal)(load_config("configs/m1_local.yaml")["infra"]["redis_url"], dim=3)
    items = embed_records(lambda ts: [[1.0, 0.0, float(i)] for i, _ in enumerate(ts)], snap,
                          {"raw_turn_user_chars": 2000, "raw_turn_assistant_chars": 600}, dim=3)
    w2.store_embeddings(ns, items)
    _, snap2 = w2.snapshot(ns)
    assert all(r["embedding"] is not None and r["sent_embeddings"].shape[1] == 3 for r in snap2)
    wal.reset(ns)
