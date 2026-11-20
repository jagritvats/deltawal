"""Per-session structured fact extraction.

Each fact carries (entity_key, fact_text, source_epoch, event_date) so that updates to the
same entity collide in the store. Extraction runs once per instance in chronological order
(the extractor sees the registry of keys created so far, encouraging key reuse on updates)
and is cached; every system then consumes the *same* facts, so systems differ only in
commit/read semantics.
"""
from __future__ import annotations

import re
import threading
from dataclasses import asdict, dataclass

from ..data.stream import PreparedInstance, StreamSession
from ..llm import OllamaClient

FACT_SCHEMA = {
    "type": "object",
    "properties": {
        # Scratchpad field decoded first: listing every user mention before writing facts
        # fixes dropped in-passing details (e.g. "they've been staying with me for nine months").
        "user_mentions": {"type": "array", "items": {"type": "string"}},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_key": {"type": "string"},
                    "fact": {"type": "string"},
                    "event_date": {"type": "string"},
                },
                "required": ["entity_key", "fact", "event_date"],
            },
        }
    },
    "required": ["user_mentions", "facts"],
}

SYSTEM_PROMPT = """You are building a long-term memory of the USER from one chat session.

First, in "user_mentions", briefly list EVERY personal detail the user states about themselves or their life in their own messages, including details mentioned in passing inside a question (e.g. "my parents have been staying with me for nine months now", "I just got back from Japan", "my 5K personal best is 27:12"). Then convert each into a fact.

Rules for facts:
- entity_key names the attribute: lower_snake segments joined by dots, e.g. "user.car.model", "user.5k_run.personal_best", "user.parents.stay_duration".
- If the attribute already has a key in EXISTING KEYS, reuse that exact key (this is how updates are tracked).
- Use a new key for genuinely different attributes; make keys specific enough that two different facts never share a key.
- fact: one self-contained sentence with concrete values (names, numbers, durations, dates).
- event_date: YYYY-MM-DD if stated or inferable from the session date, else "unknown".
- Ignore general knowledge or advice from the assistant.
- At most {max_facts} facts. If the user shares nothing personal, return empty lists."""

_KEY_RE = re.compile(r"[^a-z0-9_.]+")


def normalize_key(k: str) -> str:
    k = k.strip().lower().replace(" ", "_").replace("-", "_")
    k = _KEY_RE.sub("", k).strip(".")
    k = re.sub(r"\.{2,}", ".", k)
    return k or "user.misc"


@dataclass(frozen=True)
class Fact:
    entity_key: str
    fact_text: str
    source_epoch: int
    event_date: str
    session_date: str

    def doc_text(self) -> str:
        return f"{self.entity_key}: {self.fact_text}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionStats:
    calls: int = 0
    parse_failures: int = 0
    retries: int = 0
    wall_s: float = 0.0
    cached: int = 0
    facts: int = 0


def render_session(ss: StreamSession, user_chars: int, asst_chars: int) -> str:
    lines = [f"SESSION DATE: {ss.session.date}"]
    for t in ss.session.turns:
        lim = user_chars if t.role == "user" else asst_chars
        txt = t.content if len(t.content) <= lim else t.content[:lim] + " [...]"
        lines.append(f"[{t.role}] {txt}")
    return "\n".join(lines)


class FactExtractor:
    def __init__(self, client: OllamaClient, model: str, max_facts: int = 10, registry_max_keys: int = 150,
                 user_chars: int = 2000, asst_chars: int = 300):
        self.client = client
        self.model = model
        self.max_facts = max_facts
        self.registry_max_keys = registry_max_keys
        self.user_chars = user_chars
        self.asst_chars = asst_chars
        self._memo: dict[tuple[str, int], dict[int, list[Fact]]] = {}
        self.durations: dict[tuple[str, int], float] = {}   # (qid, epoch) -> live wall s of that extraction
        self._lock = threading.Lock()
        self.stats = ExtractionStats()

    def extract_instance(self, prep: PreparedInstance, order: list[int] | None = None) -> dict[int, list[Fact]]:
        """Extract every session. `order` (0-based stream indices) is the worker commit order:
        each session's key registry then contains ONLY keys from sessions committed before it
        (committed-only registry; realistic for reordered/concurrent workers). order=None means
        chronological, which is also what a sequential in-order worker sees."""
        if order is not None and list(order) == list(range(prep.n)):
            order = None
        memo_key = (prep.inst.qid, prep.n, tuple(order) if order is not None else None)
        with self._lock:
            if memo_key in self._memo:
                return self._memo[memo_key]
        registry: dict[str, int] = {}   # key -> processing step when last written (recency order)
        out: dict[int, list[Fact]] = {}
        sys_prompt = SYSTEM_PROMPT.format(max_facts=self.max_facts)
        seq = prep.stream if order is None else [prep.stream[i] for i in order]
        for step, ss in enumerate(seq, start=1):
            recent = sorted(registry, key=lambda k: -registry[k])[: self.registry_max_keys]
            user_msg = (
                "EXISTING KEYS:\n" + ("\n".join(sorted(recent)) if recent else "(none)") + "\n\n"
                + render_session(ss, self.user_chars, self.asst_chars)
            )
            res, attempts = self.client.chat_json(
                self.model,
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
                schema=FACT_SCHEMA, max_tokens=1200,
            )
            with self._lock:
                self.durations[(prep.inst.qid, ss.epoch)] = res.duration_s
                self.stats.calls += 1
                self.stats.retries += attempts - 1
                self.stats.cached += int(res.cached)
                self.stats.wall_s += 0.0 if res.cached else res.duration_s
                if res.parse_error:
                    self.stats.parse_failures += 1
            facts_by_key: dict[str, Fact] = {}
            items = (res.parsed or {}).get("facts", []) if isinstance(res.parsed, dict) else []
            for it in items[: self.max_facts]:
                if not isinstance(it, dict) or not str(it.get("fact", "")).strip():
                    continue
                key = normalize_key(str(it.get("entity_key", "")))
                facts_by_key[key] = Fact(
                    entity_key=key, fact_text=str(it["fact"]).strip(), source_epoch=ss.epoch,
                    event_date=str(it.get("event_date", "unknown")).strip() or "unknown",
                    session_date=ss.session.date,
                )
            out[ss.epoch] = list(facts_by_key.values())
            for k in facts_by_key:
                registry[k] = step
            with self._lock:
                self.stats.facts += len(facts_by_key)
        with self._lock:
            self._memo[memo_key] = out
        return out
