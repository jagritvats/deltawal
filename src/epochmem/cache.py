"""Deterministic SQLite cache for every LLM and embedding call.

Key = sha256 of the canonical JSON of (kind, endpoint, model, messages/input, params).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, kind TEXT, model TEXT, "
            "value TEXT NOT NULL, created REAL)"
        )
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: Any, kind: str = "", model: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, kind, model, value, created) VALUES (?,?,?,?,?)",
                (key, kind, model, json.dumps(value, ensure_ascii=False), time.time()),
            )

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
