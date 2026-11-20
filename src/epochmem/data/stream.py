"""Turn an instance into an epoch-numbered ingestion stream.

cutoff_mode:
  post_update   - stream ends right after the last evidence session, so the question
                  arrives immediately after the (possibly superseding) update. Abstention
                  questions, or any instance whose evidence ids are absent from the
                  haystack, keep the full history.
  full_haystack - the whole history is streamed.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import Instance, Session

CUTOFF_MODES = ("post_update", "full_haystack")


@dataclass(frozen=True)
class StreamSession:
    epoch: int            # monotonic, 1-based, in produce (chronological) order
    session: Session
    is_evidence: bool


@dataclass(frozen=True)
class PreparedInstance:
    inst: Instance
    stream: tuple[StreamSession, ...]
    cutoff_mode_applied: str

    @property
    def n(self) -> int:
        return len(self.stream)

    def evidence_epochs(self) -> list[int]:
        return [s.epoch for s in self.stream if s.is_evidence]

    def superseded_evidence(self, max_chars: int = 800) -> list[str]:
        """has_answer turns from every evidence session except the latest one (KU old values)."""
        ev = [s for s in self.stream if s.is_evidence]
        out = []
        for s in ev[:-1]:
            for t in s.session.turns:
                if t.has_answer:
                    out.append(f"[{s.session.date}] {t.role}: {t.content[:max_chars]}")
        return out


def prepare(inst: Instance, cutoff_mode: str) -> PreparedInstance:
    if cutoff_mode not in CUTOFF_MODES:
        raise ValueError(f"cutoff_mode must be one of {CUTOFF_MODES}")
    ev = set(inst.evidence_session_ids)
    sessions = list(inst.sessions)
    ev_idx = [i for i, s in enumerate(sessions) if s.session_id in ev]
    applied = cutoff_mode
    if cutoff_mode == "post_update":
        if inst.is_abstention or not ev_idx:
            applied = "full_haystack"
        else:
            sessions = sessions[: max(ev_idx) + 1]
    stream = tuple(
        StreamSession(epoch=i + 1, session=s, is_evidence=s.session_id in ev) for i, s in enumerate(sessions)
    )
    return PreparedInstance(inst=inst, stream=stream, cutoff_mode_applied=applied)
