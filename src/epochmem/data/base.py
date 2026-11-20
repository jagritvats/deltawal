"""Dataset-agnostic types. New benchmarks (e.g. LoCoMo in M2) implement `MemoryDataset`."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    has_answer: bool = False


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Instance:
    qid: str
    qtype: str
    question: str
    answer: str
    question_date: str
    sessions: tuple[Session, ...]                 # full chronological history
    evidence_session_ids: tuple[str, ...]
    is_abstention: bool
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def stratum(self) -> str:
        return "abstention" if self.is_abstention else self.qtype


class MemoryDataset(ABC):
    name: str = "abstract"

    @abstractmethod
    def load(self) -> list[Instance]:
        """Return every instance with sessions in chronological order."""
