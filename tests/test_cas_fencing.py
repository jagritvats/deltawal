"""CAS epoch fencing vs naive last-writer-wins under out-of-order commits (needs local Postgres)."""
from __future__ import annotations

import random
import uuid

import pytest

from epochmem.config import load_config
from epochmem.memory.extract import Fact
from epochmem.stream.schedule import barrier, commit_order
from epochmem.systems.base import compute_oocr

psycopg = pytest.importorskip("psycopg")


@pytest.fixture(scope="module")
def store():
    from epochmem.store.pg import FactStore
    cfg = load_config("configs/m1_local.yaml")
    try:
        with psycopg.connect(cfg["infra"]["pg_dsn"], autocommit=True, connect_timeout=5) as c:
            c.execute("DROP TABLE IF EXISTS facts_test")
        # dedicated 4-dim table so tests never touch experiment data
        s = FactStore(cfg["infra"]["pg_dsn"], dim=4, table="facts_test")
    except psycopg.OperationalError as e:
        pytest.skip(f"postgres unavailable: {e}")
    yield s
    s.conn.execute("DROP TABLE IF EXISTS facts_test")


def _fact(key, epoch):
    return Fact(entity_key=key, fact_text=f"{key}@{epoch}", source_epoch=epoch, event_date="unknown", session_date="d")


def _stored(store, ns):
    rows = store.conn.execute("SELECT entity_key, source_epoch FROM facts_test WHERE user_id=%s", (ns,)).fetchall()
    return {r[0]: r[1] for r in rows}


E = [0.1, 0.2, 0.3, 0.4]


def test_cas_rejects_older_epoch(store):
    ns = f"t-{uuid.uuid4()}"
    assert store.upsert(ns, _fact("user.car", 5), E, "cas") is True
    assert store.upsert(ns, _fact("user.car", 3), E, "cas") is False      # fenced
    assert store.upsert(ns, _fact("user.car", 5), E, "cas") is False      # equal epoch also fenced
    assert _stored(store, ns) == {"user.car": 5}
    assert store.upsert(ns, _fact("user.car", 7), E, "cas") is True
    assert _stored(store, ns) == {"user.car": 7}


def test_lww_overwrites_with_older_epoch(store):
    ns = f"t-{uuid.uuid4()}"
    store.upsert(ns, _fact("user.car", 5), E, "lww")
    store.upsert(ns, _fact("user.car", 3), E, "lww")
    assert _stored(store, ns) == {"user.car": 3}


def test_oocr_replay_cas_zero_lww_positive(store):
    """Replay a 40-session stream with p_ooo=0.30; every session updates a few shared keys."""
    rng = random.Random(42)
    n = 40
    facts_by_epoch = {e: [_fact(f"user.k{rng.randrange(12)}", e) for _ in range(3)] for e in range(1, n + 1)}
    order = commit_order(n, 0.30, "42:test")
    assert sorted(order) == list(range(n)) and order != list(range(n))
    applied = barrier(order, 0).applied
    res = {}
    for policy in ("lww", "cas"):
        ns = f"t-{policy}-{uuid.uuid4()}"
        applied_facts = []
        for idx in applied:
            for f in facts_by_epoch[idx + 1]:
                store.upsert(ns, f, E, policy)
                applied_facts.append(f)
        res[policy] = compute_oocr(applied_facts, _stored(store, ns))
    n_up, n_cor_cas, _ = res["cas"]
    assert n_up > 0 and n_cor_cas == 0
    assert res["lww"][1] > 0


def test_in_order_lww_has_no_corruption(store):
    ns = f"t-{uuid.uuid4()}"
    applied_facts = []
    for e in range(1, 11):
        f = _fact("user.city", e)
        store.upsert(ns, f, E, "lww")
        applied_facts.append(f)
    assert compute_oocr(applied_facts, _stored(store, ns))[1] == 0


def test_versioned_store_is_append_only():
    from epochmem.config import load_config
    from epochmem.store.pg import FactStore
    cfg = load_config("configs/m1_local.yaml")
    with psycopg.connect(cfg["infra"]["pg_dsn"], autocommit=True) as c:
        c.execute("DROP TABLE IF EXISTS facts_vtest")
    s = FactStore(cfg["infra"]["pg_dsn"], dim=4, table="facts_vtest", versioned=True)
    ns = f"t-{uuid.uuid4()}"
    for e in (5, 3, 5, 7):
        s.upsert(ns, _fact("user.car", e), E, "append")
    rows = s.conn.execute("SELECT source_epoch FROM facts_vtest WHERE user_id=%s ORDER BY 1", (ns,)).fetchall()
    assert [r[0] for r in rows] == [3, 5, 7]                 # every version kept, duplicates ignored
    assert s.key_epochs(ns) == {"user.car": 7}               # latest version per key
    s.conn.execute("DROP TABLE IF EXISTS facts_vtest")


def test_concurrent_store_init_on_fresh_table_does_not_race():
    """Regression: parallel workers creating the same fresh table used to hit a pg_type UniqueViolation."""
    from concurrent.futures import ThreadPoolExecutor
    from epochmem.config import load_config
    from epochmem.store.pg import FactStore
    cfg = load_config("configs/m1_local.yaml")
    for versioned in (False, True):
        table = f"facts_race_{uuid.uuid4().hex[:8]}"
        try:
            with ThreadPoolExecutor(8) as ex:
                stores = list(ex.map(lambda _: FactStore(cfg["infra"]["pg_dsn"], dim=4, table=table,
                                                         versioned=versioned), range(16)))
            assert len(stores) == 16
        finally:
            with psycopg.connect(cfg["infra"]["pg_dsn"], autocommit=True) as c:
                c.execute(f"DROP TABLE IF EXISTS {table}")
