"""DeltaWAL-Versioned ablations and prompt probes (additive; the systems in deltawal.py are unchanged).

Every class below is DeltaWAL-Versioned (same append-only store, same write-ahead WAL, same write path) with
exactly one read-path factor changed:

  deltawal_versioned_rerun         : identical prompt; the local reader is called WITHOUT the response cache
                                     (reproducibility check of a cached greedy run; cloud readers stay cached)
  deltawal_versioned_pad           : meant for lag 0; the top-k fact list is extended with the next-ranked stored
                                     facts until the prompt reaches a per-question token target (the same
                                     question's lag-1 prompt length), i.e. a long prompt without a WAL tail
  deltawal_versioned_sentdate      : WAL-tail turns tagged "message sent on <session date>" and relative date
                                     phrases in them ("yesterday", "3 weeks ago", ...) resolved against that date
  deltawal_versioned_noprec        : no epoch-precedence paragraph in the system prompt
  deltawal_versioned_tailfirst     : gated tail rendered before the facts instead of in epoch order
  deltawal_versioned_sentdate_tailfirst : sentdate + tailfirst together (follow-up probe)
  deltawal_versioned_notags        : no epoch / OLDER-NEWER tags (lines keep their date and their epoch order)
  deltawal_versioned_nogate        : no relevance gate: the whole uncommitted tail (all roles), newest turns kept
                                     until the whole prompt reaches `nogate_token_budget`
  deltawal_versioned_recency_user  : the gate replaced by B3-style structural recency: trailing USER turns of the
                                     conversation (watermark-agnostic) up to `recency_token_budget`, rendered
                                     with the same epoch tags and precedence instruction
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time

from ..config import repo_path
from ..llm import OllamaClient, provider_of
from ..tokens import count_messages
from .base import READER_SYSTEM, SystemOutput
from .deltawal import DeltaWALVersioned, strip_emb


# ---------------------------------------------------------------- relative-date resolution (sentdate probe)
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "a couple": 2, "couple of": 2}
_WD = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUMRE = r"(\d+|a couple of|a couple|couple of|an|a|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_PATTERNS = [
    (re.compile(r"\bthe day before yesterday\b", re.I), "dby"),
    (re.compile(r"\byesterday\b", re.I), "yesterday"),
    (re.compile(r"\b(today|tonight|this morning|this afternoon|this evening)\b", re.I), "today"),
    (re.compile(r"\btomorrow\b", re.I), "tomorrow"),
    (re.compile(rf"\b{_NUMRE}\s+(day|week|month|year)s?\s+ago\b", re.I), "ago"),
    (re.compile(r"\blast\s+(week|weekend|month|year)\b", re.I), "last_unit"),
    (re.compile(r"\b(?:last|this past)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.I), "last_wd"),
]


def _parse_date(s: str) -> dt.date | None:
    m = re.match(r"\s*(\d{4})[/-](\d{1,2})[/-](\d{1,2})", s or "")
    return dt.date(int(m[1]), int(m[2]), int(m[3])) if m else None


def _minus_months(d: dt.date, n: int) -> dt.date:
    y, mth = divmod(d.year * 12 + d.month - 1 - n, 12)
    return dt.date(y, mth + 1, 1)


def _fmt(d: dt.date) -> str:
    return d.strftime("%Y/%m/%d")


def _resolve(kind: str, m: re.Match, d: dt.date) -> str | None:
    if kind == "dby":
        return _fmt(d - dt.timedelta(days=2))
    if kind == "yesterday":
        return _fmt(d - dt.timedelta(days=1))
    if kind == "today":
        return _fmt(d)
    if kind == "tomorrow":
        return _fmt(d + dt.timedelta(days=1))
    if kind == "ago":
        raw, unit = m.group(1).lower(), m.group(2).lower()
        n = int(raw) if raw.isdigit() else _NUM.get(raw)
        if not n:
            return None
        if unit == "day":
            return _fmt(d - dt.timedelta(days=n))
        if unit == "week":
            return "around " + _fmt(d - dt.timedelta(weeks=n))
        if unit == "month":
            return "around " + _minus_months(d, n).strftime("%Y/%m")
        return "around " + str(d.year - n)
    if kind == "last_unit":
        unit = m.group(1).lower()
        if unit == "week":
            start = d - dt.timedelta(days=d.weekday() + 7)
            return "week of " + _fmt(start)
        if unit == "weekend":
            sat = d - dt.timedelta(days=(d.weekday() - 5) % 7 or 7)
            return "weekend of " + _fmt(sat)
        if unit == "month":
            return _minus_months(d, 1).strftime("%Y/%m")
        return str(d.year - 1)
    if kind == "last_wd":
        wd = _WD.index(m.group(1).lower())
        back = (d.weekday() - wd) % 7 or 7
        return _fmt(d - dt.timedelta(days=back))
    return None


def resolve_relative_dates(text: str, anchor: str) -> tuple[str, int]:
    """Append '[= <absolute date>]' after each relative date phrase, resolved against `anchor`
    (a LongMemEval session date such as '2022/11/18 (Fri) 17:03'). Returns (text, n_resolved)."""
    d = _parse_date(anchor)
    if d is None:
        return text, 0
    spans = []
    for rx, kind in _PATTERNS:
        for m in rx.finditer(text):
            if any(a < m.end() and m.start() < b for a, b, _ in spans):
                continue                      # overlaps an earlier (more specific) match
            r = _resolve(kind, m, d)
            if r:
                spans.append((m.start(), m.end(), r))
    out, last = [], 0
    for a, b, r in sorted(spans):
        out.append(text[last:b] + f" [= {r}]")
        last = b
    out.append(text[last:])
    return "".join(out), len(spans)


# ---------------------------------------------------------------- systems
_LIVE: dict = {}


class DeltaWALVersionedRerun(DeltaWALVersioned):
    name = "deltawal_versioned_rerun"

    def _live_client(self):
        local = self.ctx.client.local if hasattr(self.ctx.client, "local") else self.ctx.client
        key = id(local)
        if key not in _LIVE:
            d = local.defaults
            _LIVE[key] = OllamaClient(local.base_url, None, num_ctx=d["num_ctx"], temperature=d["temperature"],
                                      seed=d["seed"], think=local.think)
        return _LIVE[key]

    def run(self, prep, ns, lag, p_ooo):
        ing = self.ingest(prep, ns, lag, p_ooo)
        qemb = self.ctx.embed_query(prep.inst.question)
        self.read_diag, self.read_llm_ms = {}, 0.0
        t0 = time.perf_counter()
        messages, retrieved, extra = self.build_messages(prep, ns, ing, qemb)
        read_ms = 1000 * (time.perf_counter() - t0)
        model = self.ctx.m["reader"]
        client = self._live_client() if provider_of(model) is None else self.ctx.client
        res = client.chat(model, messages, max_tokens=256)
        self.read_diag["reader_uncached"] = provider_of(model) is None
        return SystemOutput(
            answer=res.content.strip(), messages=messages, prompt_tokens=count_messages(messages),
            server_prompt_tokens=res.prompt_eval_count, retrieved=retrieved, extra_context=extra,
            ingest=ing, reader_cached=res.cached, read_path_ms=round(read_ms, 3),
            read_llm_ms=round(self.read_llm_ms, 1), read_diag=self.read_diag)


_PAD_TARGETS: dict = {}


def _pad_target(cfg, qid: str) -> int:
    s = cfg["systems"]
    path = s.get("pad_targets_file")
    if path and path not in _PAD_TARGETS:
        _PAD_TARGETS[path] = json.loads(repo_path(path).read_text(encoding="utf-8"))
    tgt = (_PAD_TARGETS.get(path) or {}).get(qid) if path else None
    return int(tgt if tgt is not None else s.get("pad_token_budget", 1255))


class DeltaWALVersionedPad(DeltaWALVersioned):
    name = "deltawal_versioned_pad"

    def build_messages(self, prep, ns, ing, qemb):
        facts, gated = self._common(ns, qemb)
        target = _pad_target(self.ctx.cfg, prep.inst.qid)
        pool = self.store.search(ns, qemb, int(self.ctx.cfg["systems"].get("pad_max_facts", 200)))
        seen = {(f["entity_key"], f["source_epoch"], f["fact_text"]) for f in facts}
        extra_facts = [f for f in pool if (f["entity_key"], f["source_epoch"], f["fact_text"]) not in seen]
        cur = list(facts)
        msgs = self._assemble(prep, cur, gated)
        tok = count_messages(msgs)
        for f in extra_facts:
            if tok >= target:
                break
            nxt = cur + [f]
            nmsgs = self._assemble(prep, nxt, gated)
            ntok = count_messages(nmsgs)
            if ntok > target and (ntok - target) > (target - tok):
                break                           # adding this fact overshoots more than stopping undershoots
            cur, msgs, tok = nxt, nmsgs, ntok
        self.read_diag.update({"pad_target_tokens": target, "pad_n_facts": len(cur), "pad_added": len(cur) - len(facts)})
        extra = [{"epoch": t["epoch"], "turn": t["turn"], "role": t["role"], "sim": t["sim"]} for t in gated]
        return msgs, [strip_emb(f) for f in cur], extra


class DeltaWALVersionedSentDate(DeltaWALVersioned):
    name = "deltawal_versioned_sentdate"

    def _turn_line(self, t):
        text, n = resolve_relative_dates(self.raw_text(t["role"], t["content"]), t["date"])
        self.read_diag["dates_resolved"] = self.read_diag.get("dates_resolved", 0) + n
        return f"[NEWER MESSAGE | epoch={t['epoch']} | message sent on {t['date']}] [{t['role']}] {text}"


class DeltaWALVersionedNoPrec(DeltaWALVersioned):
    name = "deltawal_versioned_noprec"

    def _messages(self, prep, lines):
        user = "Memory:\n" + ("\n".join(lines) or "(none)") + f"\n\n{self.question_block(prep)}"
        return [{"role": "system", "content": READER_SYSTEM}, {"role": "user", "content": user}]


class DeltaWALVersionedTailFirst(DeltaWALVersioned):
    name = "deltawal_versioned_tailfirst"

    def _assemble(self, prep, facts, turns):
        fl = [self._fact_line(f) for f in sorted(facts, key=lambda f: f["source_epoch"])]
        tl = [self._turn_line(t) for t in sorted(turns, key=lambda t: (t["epoch"], t["turn"]))]
        return self._messages(prep, tl + fl)


class DeltaWALVersionedSentDateTailFirst(DeltaWALVersionedSentDate):
    """Both temporal probes at once: sent-date tail rendering + tail before the facts (follow-up probe)."""
    name = "deltawal_versioned_sentdate_tailfirst"
    _assemble = DeltaWALVersionedTailFirst._assemble


class DeltaWALVersionedNoTags(DeltaWALVersioned):
    name = "deltawal_versioned_notags"

    def _fact_line(self, f):
        return f"[{f['session_date']}] {f['entity_key']}: {f['fact_text']}"

    def _turn_line(self, t):
        return f"[{t['date']}] {self.raw_turn(t['role'], t['content'])}"


class _BudgetTail(DeltaWALVersioned):
    budget_key = ""

    def _fill(self, prep, facts, turns):
        budget = int(self.ctx.cfg["systems"][self.budget_key])
        tail: list = []
        for t in reversed(turns):               # newest first; stop at the first turn that does not fit
            cand = [t] + tail
            if count_messages(self._assemble(prep, facts, cand)) > budget:
                break
            tail = cand
        return tail


class DeltaWALVersionedNoGate(_BudgetTail):
    name = "deltawal_versioned_nogate"
    budget_key = "nogate_token_budget"

    def build_messages(self, prep, ns, ing, qemb):
        facts = self.store.search(ns, qemb, self.ctx.cfg["memory"]["top_k_facts"])
        committed, delta = self.ctx.wal.snapshot(ns)
        tail = self._fill(prep, facts, delta)
        self.read_diag.update({"committed_epoch": committed, "n_uncommitted_turns": len(delta),
                               "b3_tail_turns": len(tail), "tail": [(t["epoch"], t["turn"]) for t in tail]})
        extra = [{"epoch": t["epoch"], "turn": t["turn"], "role": t["role"]} for t in tail]
        return self._assemble(prep, facts, tail), [strip_emb(f) for f in facts], extra


class DeltaWALVersionedRecencyUser(_BudgetTail):
    name = "deltawal_versioned_recency_user"
    budget_key = "recency_token_budget"

    def build_messages(self, prep, ns, ing, qemb):
        facts = self.store.search(ns, qemb, self.ctx.cfg["memory"]["top_k_facts"])
        turns = [{"epoch": ss.epoch, "turn": i, "role": t.role, "content": t.content, "date": ss.session.date}
                 for ss in prep.stream for i, t in enumerate(ss.session.turns) if t.role == "user"]
        tail = self._fill(prep, facts, turns)
        self.read_diag.update({"b3_tail_turns": len(tail), "tail": [(t["epoch"], t["turn"]) for t in tail],
                               "inflight_turns_in_tail": sum(t["epoch"] in set(ing.inflight_epochs) for t in tail)
                               if ing is not None else None})
        extra = [{"epoch": t["epoch"], "turn": t["turn"], "role": t["role"]} for t in tail]
        return self._assemble(prep, facts, tail), [strip_emb(f) for f in facts], extra


ABLATION_SYSTEMS = (DeltaWALVersionedRerun, DeltaWALVersionedPad, DeltaWALVersionedSentDate, DeltaWALVersionedNoPrec,
                    DeltaWALVersionedTailFirst, DeltaWALVersionedSentDateTailFirst, DeltaWALVersionedNoTags, DeltaWALVersionedNoGate,
                    DeltaWALVersionedRecencyUser)


# ---------------------------------------------------------------- B3 + dates (isolates session dates on raw turns)
from .b3_recency import B3RecencyMatchedUser  # noqa: E402


class B3RecencyMatchedUserDated(B3RecencyMatchedUser):
    """Exactly B3_recency_matched_user (same LWW async index, same `b3_matched_token_budget`, same plain fact list,
    no epoch tags, no precedence instruction), except that each raw turn is prefixed with its session date."""
    name = "B3_recency_matched_user_dated"

    def build_messages(self, prep, ns, ing, qemb):
        facts = self.retrieve_facts(ns, qemb)
        budget = self.ctx.cfg["systems"]["b3_matched_token_budget"]
        mem = "Retrieved memory facts:\n" + ("\n".join(map(self.fmt_fact_plain, facts)) or "(none)")
        all_turns = [(ss.session.date, t) for ss in prep.stream for t in ss.session.turns
                     if not self.user_only or t.role == "user"]

        def msgs_for(tail):
            recent = "Recent conversation:\n" + "\n".join(f"[{d}] {self.raw_turn(t.role, t.content)}" for d, t in tail)
            body = f"{mem}\n\n{recent}\n\n{self.question_block(prep)}" if tail else f"{mem}\n\n{self.question_block(prep)}"
            return [{"role": "system", "content": READER_SYSTEM}, {"role": "user", "content": body}]

        tail: list = []
        for dt_t in reversed(all_turns):        # newest first; stop at the first turn that does not fit
            cand = [dt_t] + tail
            if count_messages(msgs_for(cand)) > budget:
                break
            tail = cand
        self.read_diag["b3_tail_turns"] = len(tail)
        return msgs_for(tail), facts, [{"role": t.role, "content": t.content} for _, t in tail]


ABLATION_SYSTEMS = ABLATION_SYSTEMS + (B3RecencyMatchedUserDated,)
