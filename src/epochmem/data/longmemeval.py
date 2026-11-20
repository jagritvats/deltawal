"""LongMemEval-S (cleaned) loader. Refuses the oracle split by construction."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from .base import Instance, MemoryDataset, Session, Turn

EXPECTED_N = 500


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def _parse_date(s: str) -> datetime:
    # "2023/05/20 (Sat) 02:21"
    return datetime.strptime(s.split(" (")[0] + " " + s.split(") ")[-1], "%Y/%m/%d %H:%M")


class LongMemEvalDataset(MemoryDataset):
    name = "longmemeval_s"

    def __init__(self, path: str | Path, expected_sha256: str | None = None, expected_n: int | None = EXPECTED_N):
        self.path = Path(path)
        self.expected_n = expected_n
        if "oracle" in self.path.name.lower():
            raise ValueError("The oracle split is not allowed; use longmemeval_s_cleaned.json.")
        if "s_cleaned" not in self.path.name:
            raise ValueError(f"Expected longmemeval_s_cleaned.json, got {self.path.name}")
        self.expected_sha256 = expected_sha256

    def verify(self) -> str:
        digest = sha256_file(self.path)
        if self.expected_sha256 and digest != self.expected_sha256:
            raise ValueError(f"sha256 mismatch: {digest} != {self.expected_sha256}")
        return digest

    def load(self) -> list[Instance]:
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if self.expected_n is not None and len(raw) != self.expected_n:
            raise ValueError(f"expected {self.expected_n} questions, found {len(raw)}")
        out = []
        for r in raw:
            sessions = []
            for sid, date, turns in zip(r["haystack_session_ids"], r["haystack_dates"], r["haystack_sessions"]):
                sessions.append(Session(
                    session_id=sid, date=date,
                    turns=tuple(Turn(t["role"], t["content"], bool(t.get("has_answer", False))) for t in turns),
                ))
            # Guarantee chronological order (stable on ties).
            order = sorted(range(len(sessions)), key=lambda i: (_parse_date(sessions[i].date), i))
            sessions = [sessions[i] for i in order]
            qid = r["question_id"]
            out.append(Instance(
                qid=qid, qtype=r["question_type"], question=r["question"], answer=str(r["answer"]),
                question_date=r["question_date"], sessions=tuple(sessions),
                evidence_session_ids=tuple(r.get("answer_session_ids") or ()),
                is_abstention=qid.endswith("_abs"),
            ))
        return out
