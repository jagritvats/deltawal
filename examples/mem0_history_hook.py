"""Read-only Mem0 history hook: put superseded values back in front of the reader, in epoch order.

Mem0 OSS 1.0.11 resolves conflicts with an in-place UPDATE. Search then returns only the newest text of each
memory, so "what did I have before?" questions fail, even though Memory.history() still holds every earlier
version. This wrapper changes nothing in Mem0. For each memory that search returns, it reads the memory's
history and renders the earlier ADD/UPDATE texts, oldest first, before the current value.

Mirrors scripts/run_mem0_history_hook.py; in the paper it lifts KU accuracy on Mem0 from 76.7% to 88.3-91.7%
across three readers, with no losses (paper/numbers_manifest.json: tab4, paired/mem0_hook_vs_sync_ku).

  uv run python examples/mem0_history_hook.py --dry-run     # stub Memory, no LLM or API key
Real use:
  from mem0 import Memory
  m = Memory.from_config(cfg)            # tag writes: m.add(msgs, user_id=u, metadata={"source_epoch": e, "session_date": d})
  lines = search_with_history(m, "Where did I live before Porto?", user_id=u)
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from typing import Any, ClassVar


def _epoch(md: dict | None) -> int:
    e = (md or {}).get("source_epoch")
    return int(e) if e is not None else -1


def search_with_history(memory: Any, query: str, user_id: str, limit: int = 12,
                        epoch_of_version: Callable[[dict, dict], int | None] | None = None) -> list[str]:
    """Return reader lines for `query`: current memories plus their earlier versions, sorted by epoch.

    Mem0's history rows carry wall-clock timestamps, not epochs. By default an earlier version inherits no
    epoch and sorts just before its memory's current value. Pass `epoch_of_version(memory_hit, history_row)`
    if you log epochs at write time; the paper maps them from its ingestion event log.
    """
    items: list[tuple[int, int, int, str]] = []
    for rank, hit in enumerate(memory.search(query, user_id=user_id, limit=limit).get("results", [])):
        md = hit.get("metadata") or {}
        cur_epoch = _epoch(md)
        versions = [h for h in memory.history(hit["id"]) if h["event"] in ("ADD", "UPDATE")]
        for v in versions[:-1]:                         # Mem0 orders by created_at; the last version is the current value
            e = epoch_of_version(hit, v) if epoch_of_version else None
            when = (v.get("updated_at") or v.get("created_at") or "")[:10]
            items.append(((e if e is not None else cur_epoch), 0, rank,
                          f"[MEMORY (earlier version) | epoch={e if e is not None else '?'} | {when}] {v['new_memory']}"))
        items.append((cur_epoch, 1, rank,
                      f"[MEMORY | epoch={md.get('source_epoch')} | {md.get('session_date', '')}] {hit['memory']}"))
    return [x[3] for x in sorted(items, key=lambda x: (x[0], x[1], x[2]))]


class StubMemory:
    """Mimics the two Mem0 1.0.11 calls the hook uses (search result and history row shapes)."""

    _HIST: ClassVar[dict] = {"m1": [
        {"event": "ADD", "new_memory": "Lives in Lisbon", "old_memory": None, "created_at": "2026-03-02T10:00:00"},
        {"event": "UPDATE", "new_memory": "Lives in Porto", "old_memory": "Lives in Lisbon",
         "created_at": "2026-03-02T10:00:00", "updated_at": "2026-06-21T18:30:00"}],
        "m2": [{"event": "ADD", "new_memory": "Takes a pottery class on weekends", "old_memory": None,
                "created_at": "2026-05-10T09:00:00"}]}
    _EPOCH: ClassVar[dict] = {"Lives in Lisbon": 1, "Lives in Porto": 3}

    def search(self, query, *, user_id=None, limit=100, **_):
        return {"results": [
            {"id": "m1", "memory": "Lives in Porto", "score": 0.81,
             "metadata": {"source_epoch": 3, "session_date": "2026-06-21"}},
            {"id": "m2", "memory": "Takes a pottery class on weekends", "score": 0.22,
             "metadata": {"source_epoch": 2, "session_date": "2026-05-10"}}][:limit]}

    def history(self, memory_id):
        return self._HIST.get(memory_id, [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="use a stub Memory (no Mem0 install, LLM or key needed)")
    args = ap.parse_args()
    if not args.dry_run:
        ap.error("this demo only ships the --dry-run path; see the module docstring for real use")
    m, q = StubMemory(), "Which city did I live in before I moved to Porto?"
    plain = [f"[MEMORY | epoch={h['metadata']['source_epoch']}] {h['memory']}" for h in m.search(q, user_id="u")["results"]]
    print("Mem0 search alone (Lisbon is gone):\n  " + "\n  ".join(plain))
    lines = search_with_history(m, q, user_id="u",
                                epoch_of_version=lambda hit, v: StubMemory._EPOCH.get(v["new_memory"]))
    print("\nwith the read-only history hook:\n  " + "\n  ".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
