"""PostgreSQL + pgvector fact store keyed by (user_id, entity_key).

Two write policies:
  lww - naive last-writer-wins by arrival order (unconditional overwrite).
  cas - monotonic epoch fencing: an upsert only lands if its source_epoch is newer.
"""
from __future__ import annotations

import threading

import psycopg

from ..memory.extract import Fact

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS {table} (
    user_id      TEXT NOT NULL,
    entity_key   TEXT NOT NULL,
    fact_text    TEXT NOT NULL,
    source_epoch INTEGER NOT NULL,
    event_date   TEXT,
    session_date TEXT,
    embedding    vector({dim}) NOT NULL,
    PRIMARY KEY (user_id, entity_key)
);
"""

UPSERT_LWW = """
INSERT INTO {table} (user_id, entity_key, fact_text, source_epoch, event_date, session_date, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
ON CONFLICT (user_id, entity_key) DO UPDATE SET
    fact_text = EXCLUDED.fact_text, source_epoch = EXCLUDED.source_epoch,
    event_date = EXCLUDED.event_date, session_date = EXCLUDED.session_date,
    embedding = EXCLUDED.embedding
"""

# Monotonic fence: ON CONFLICT (user_id, entity_key) DO UPDATE ... WHERE EXCLUDED.source_epoch > facts.source_epoch
UPSERT_CAS = UPSERT_LWW + "\nWHERE EXCLUDED.source_epoch > {table}.source_epoch"

POLICIES = {"lww": UPSERT_LWW, "cas": UPSERT_CAS}


def vec_literal(v: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v) + "]"


class FactStore:
    def __init__(self, dsn: str, dim: int, table: str = "facts"):
        if not table.isidentifier():
            raise ValueError(table)
        self.dsn = dsn
        self.dim = dim
        self.table = table
        self._sql = {k: v.format(table=table) for k, v in POLICIES.items()}
        self._local = threading.local()
        with psycopg.connect(dsn, autocommit=True) as c:
            c.execute(SCHEMA.format(dim=dim, table=table))

    @property
    def conn(self) -> psycopg.Connection:
        c = getattr(self._local, "conn", None)
        if c is None or c.closed:
            c = psycopg.connect(self.dsn, autocommit=True)
            self._local.conn = c
        return c

    def reset(self, user_id: str) -> None:
        self.conn.execute(f"DELETE FROM {self.table} WHERE user_id = %s", (user_id,))

    def upsert(self, user_id: str, fact: Fact, emb: list[float], policy: str) -> bool:
        cur = self.conn.execute(
            self._sql[policy],
            (user_id, fact.entity_key, fact.fact_text, fact.source_epoch, fact.event_date,
             fact.session_date, vec_literal(emb)),
        )
        return cur.rowcount == 1  # False when the CAS fence rejected the write

    def search(self, user_id: str, qemb: list[float], k: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT entity_key, fact_text, source_epoch, event_date, session_date, "
            f"embedding <=> %s::vector AS dist FROM {self.table} WHERE user_id = %s ORDER BY dist LIMIT %s",
            (vec_literal(qemb), user_id, k),
        ).fetchall()
        return [dict(entity_key=r[0], fact_text=r[1], source_epoch=r[2], event_date=r[3],
                     session_date=r[4], dist=float(r[5])) for r in rows]

    def key_epochs(self, user_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            f"SELECT entity_key, source_epoch FROM {self.table} WHERE user_id = %s", (user_id,)
        ).fetchall()
        return {r[0]: r[1] for r in rows}
