"""MemConflict loader (arXiv:2605.20926; GitHub TaoZhen1110/MemConflict, file Data/Step4_4.jsonl).

Source layout: one JSON line per simulated USER (30 users). Each user has a `Full_Session_Chain`
of ~53 dated sessions (`Session_ID` 0..n-1, `Date` YYYY-MM-DD, non-decreasing). A session's
`Session_Dialogue` is a dict `dialogue_turn_<i>` -> [ {role, content}, ... ]. Questions are attached
to the session that TRIGGERS them (`Session_Questions`) and are asked right after that session.

Mapping to `Instance` (one per question):
  * qid           = "mc-<user id first 8 hex>-s<session:02d>-<question_id>"  (question ids restart
                    per session in the source, so the session index is part of the key)
  * sessions      = sessions 0..k of that user, where k is the trigger session. Later sessions are the
                    question's future (they may update the answer again) and are never included, so
                    `post_update` and `full_haystack` cut-offs coincide for this dataset.
  * session_id    = "<user8>-S<session:02d>", date = the source `Date` (YYYY-MM-DD, no time of day)
  * turns         = every message of every `dialogue_turn_<i>` in numeric order; a handful of
                    malformed messages in the release ({"assistant": "content", "content": ...},
                    {"assistant": {"content": ...}}, {"role": ...} without content) are repaired or
                    dropped. `has_answer` is always False (the source does not mark answer turns).
  * qtype         = "memconflict-dynamic" | "memconflict-static" | "memconflict-conditional"
  * question_date = date of the trigger session
  * evidence_session_ids: rebuilt from the construction metadata (the source question rows carry
                    no conflict id, but Step4_4 generates them positionally from the trigger session's
                    metadata, see `align_questions`):
        dynamic     -> [session that established the `Before` value, trigger session]
        static      -> [session holding Point_A (true value), trigger session holding the false Point_B]
        conditional -> [session holding the previous point (rank-1) whose rule is asked, trigger session]
                    If alignment fails the evidence is just the trigger session.
  * is_abstention = False (the release has no unanswerable questions).
  * meta          = raw question row, aligned conflict record, user id, trigger index, persona seed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .base import Instance, MemoryDataset, Session, Turn

QTYPE = {
    "dynamic_conflict": "memconflict-dynamic",
    "static_conflict": "memconflict-static",
    "conditional_conflict": "memconflict-conditional",
}
STRATA = tuple(QTYPE.values())

_TURN_RE = re.compile(r"(\d+)$")
_CAREER_STABLE = ("Employment_Status", "Savings_Amount", "Company_Type")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def _message(m) -> Turn | None:
    if not isinstance(m, dict):
        return None
    role = m.get("role")
    content = m.get("content")
    if role not in ("user", "assistant"):
        role = next((k for k in ("user", "assistant") if k in m), None)
    if isinstance(content, dict):
        content = content.get("content")
    if content is None and role in m and isinstance(m[role], dict):
        content = m[role].get("content")
    if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
        return None
    return Turn(role, content)


def parse_dialogue(dialogue) -> tuple[Turn, ...]:
    if isinstance(dialogue, list):
        items = [("0", dialogue)]
    elif isinstance(dialogue, dict):
        items = sorted(dialogue.items(), key=lambda kv: int(m.group(1)) if (m := _TURN_RE.search(kv[0])) else 0)
    else:
        return ()
    out = []
    for _, msgs in items:
        for m in msgs if isinstance(msgs, list) else [msgs]:
            t = _message(m)
            if t is not None:
                out.append(t)
    return tuple(out)


def _non_empty_after(u) -> bool:
    return isinstance(u, dict) and "After" in u and u.get("After") not in (None, "", {})


def _role_rank(role) -> int:
    if isinstance(role, str) and role.startswith("Point_") and len(role) >= 7 and "A" <= role[-1].upper() <= "Z":
        return ord(role[-1].upper()) - 64
    return -1


def _n_dynamic_questions(u: dict) -> int:
    """Number of template questions Step4_4.generate_dynamic_questions emits for one update."""
    before, after = u.get("Before"), u.get("After")
    if u.get("Attribute") == "Career_Status" and isinstance(before, dict) and isinstance(after, dict):
        keys = set(before) | set(after)
        changed = sum(1 for k in keys if before.get(k) != after.get(k))
        stable = any(k in keys and before.get(k) == after.get(k) for k in _CAREER_STABLE)
        return 1 + min(3, changed) + int(stable)
    return 2


def align_questions(session: dict) -> list[dict] | None:
    """Recover, for each question of a trigger session, the conflict record it was generated from.

    Step4_4 emits questions in a fixed order (dynamic per update item, then one static per Point_B+,
    then one conditional per Point_B+); an optional LLM rewrite must keep count and ids. Returns None
    if the reconstructed layout does not match the stored questions."""
    recs: list[dict] = []
    for u in session.get("Updated_Attributes") or []:
        if _non_empty_after(u):
            recs += [{"kind": "dynamic_conflict", "update": u}] * _n_dynamic_questions(u)
    for p in session.get("Static_Conflict_Information") or []:
        if isinstance(p, dict) and str(p.get("Role", "")).strip() not in ("Point_A", "Distractor", ""):
            recs.append({"kind": "static_conflict", "point": p})
    for p in session.get("Conditional_Conflict_Information") or []:
        if isinstance(p, dict) and str(p.get("Role", "")).strip() not in ("Point_A", "Distractor", ""):
            recs.append({"kind": "conditional_conflict", "point": p})
    qs = session.get("Session_Questions") or []
    if len(recs) != len(qs) or any(r["kind"] != q.get("conflict_type") for r, q in zip(recs, qs)):
        return None
    return recs


def _revealed(session: dict, attr) -> bool:
    """`attr` revealed in an initial_reveal session, top-level or one level down
    (e.g. Social_Status lives under Social_Relationships)."""
    rev = session.get("Revealed_Attributes") or {}
    return attr in rev or any(isinstance(v, dict) and attr in v for v in rev.values())


def evidence_indices(chain: list[dict], k: int, rec: dict | None) -> tuple[list[int], dict]:
    """Chain indices of the evidence sessions (chronological) plus the resolved source record."""
    if rec is None:
        return [k], {}
    if rec["kind"] == "dynamic_conflict":
        attr = rec["update"].get("Attribute")
        for j in range(k - 1, -1, -1):
            s = chain[j]
            if any(isinstance(u, dict) and u.get("Attribute") == attr and _non_empty_after(u)
                   for u in s.get("Updated_Attributes") or []) or _revealed(s, attr):
                return [j, k], {"source_session": j}
        return [k], {}
    field = "Static_Conflict_Information" if rec["kind"] == "static_conflict" else "Conditional_Conflict_Information"
    cid = rec["point"].get("Conflict_ID")
    want = 1 if rec["kind"] == "static_conflict" else _role_rank(rec["point"].get("Role")) - 1
    hit = None
    for j in range(k + 1):
        for p in chain[j].get(field) or []:
            if isinstance(p, dict) and p.get("Conflict_ID") == cid and _role_rank(p.get("Role")) == want:
                if rec["kind"] == "static_conflict" and p.get("Target_Field_Path") != rec["point"].get("Target_Field_Path"):
                    continue
                if hit is None or rec["kind"] == "conditional_conflict":
                    hit = (j, p)      # static: first Point_A wins (find_point_a_static_value);
                                      # conditional: last match wins (find_previous_conditional_point)
    if hit is None:
        return [k], {}
    return sorted({hit[0], k}), {"source_session": hit[0], "source_point": hit[1]}


FILTER_RULE = (
    "drop static questions about the user's own Name unless a USER turn of the Point_A session "
    "self-introduces with that name ('I'm|I am|my name is|name is|call me <first name>'); "
    "everything else kept"
)
_SELF_INTRO = r"(?:my name|i'm|i’m|i am|call me|name's|name is)\W+(?:is\W+)?"


def passes_quality_filter(inst: Instance) -> bool:
    """Deterministic label-quality filter (see FILTER_RULE). It targets the one systematic
    construction error found in the audit: the user's own name (static Point_A) rendered by the
    dialogue generator as a third party ("Jackson Andrews is my little sports fan")."""
    if inst.qtype != QTYPE["static_conflict"]:
        return True
    sp = (inst.meta.get("evidence_source") or {}).get("source_point") or {}
    if sp.get("Target_Field_Path") != "Name":
        return True
    name = str(sp.get("Value", "")).strip()
    if not name or len(inst.evidence_session_ids) < 2:
        return False
    src = next(s for s in inst.sessions if s.session_id == inst.evidence_session_ids[0])
    pat = re.compile(_SELF_INTRO + re.escape(name.split()[0]), re.I)
    return any(t.role == "user" and pat.search(t.content) for t in src.turns)


def user_sessions(user: dict) -> tuple[str, list[Session]]:
    uid8 = str(user["ID"]).replace("-", "")[:8]
    chain = user["Full_Session_Chain"]
    out = [Session(session_id=f"{uid8}-S{int(s.get('Session_ID', i)):02d}", date=str(s["Date"]),
                   turns=parse_dialogue(s.get("Session_Dialogue"))) for i, s in enumerate(chain)]
    return uid8, out


def user_instances(user: dict) -> list[Instance]:
    chain = user["Full_Session_Chain"]
    ids = [s.get("Session_ID", i) for i, s in enumerate(chain)]
    if ids != list(range(len(chain))):
        raise ValueError(f"user {user.get('ID')}: Session_ID is not 0..n-1 in order")
    dates = [str(s["Date"]) for s in chain]
    if dates != sorted(dates):
        raise ValueError(f"user {user.get('ID')}: session dates are not chronological")
    uid8, sessions = user_sessions(user)
    out = []
    for k, s in enumerate(chain):
        qs = s.get("Session_Questions") or []
        if not qs:
            continue
        recs = align_questions(s)
        for n, q in enumerate(qs):
            rec = recs[n] if recs is not None else None
            ev, src = evidence_indices(chain, k, rec)
            ctype = q["conflict_type"]
            out.append(Instance(
                qid=f"mc-{uid8}-s{k:02d}-{q['question_id']}",
                qtype=QTYPE[ctype], question=str(q["question"]), answer=str(q["answer"]),
                question_date=str(s["Date"]), sessions=tuple(sessions[: k + 1]),
                evidence_session_ids=tuple(sessions[j].session_id for j in ev),
                is_abstention=False,
                meta={"user_id": user["ID"], "trigger_session": k, "question_id": q["question_id"],
                      "conflict_type": ctype, "ability_target": q.get("ability_target"),
                      "difficulty": q.get("difficulty"), "aligned": recs is not None,
                      "conflict": rec, "evidence_source": src, "raw_question": q,
                      "persona_seed": (user.get("metadata") or {}).get("persona_seed")},
            ))
    return out


class MemConflictDataset(MemoryDataset):
    name = "memconflict"

    def __init__(self, path: str | Path, expected_sha256: str | None = None, qids: list[str] | None = None):
        self.path = Path(path)
        self.expected_sha256 = expected_sha256
        self.qids = list(qids) if qids is not None else None

    def verify(self) -> str:
        digest = sha256_file(self.path)
        if self.expected_sha256 and digest != self.expected_sha256:
            raise ValueError(f"sha256 mismatch: {digest} != {self.expected_sha256}")
        return digest

    def users(self) -> list[dict]:
        with self.path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def load(self) -> list[Instance]:
        if self.expected_sha256:
            self.verify()
        out = [i for u in self.users() for i in user_instances(u)]
        if len({i.qid for i in out}) != len(out):
            raise ValueError("duplicate qids")
        if self.qids is not None:
            by = {i.qid: i for i in out}
            missing = [q for q in self.qids if q not in by]
            if missing:
                raise ValueError(f"{len(missing)} requested qids not found, e.g. {missing[:3]}")
            out = [by[q] for q in self.qids]
        return out
