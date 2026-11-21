"""MemConflict (Data/Step4_4.jsonl): 30-item label audit report + stratified 90-question eval split.

No model-API calls. The audit verdicts below were assigned by an LLM assistant (not a human) reading the evidence sessions of each sampled
question (the gold answer is generated from construction metadata, so the check is whether the
dialogue actually states it for the USER). The script re-derives the sample deterministically,
re-checks structure, evaluates the candidate filter rules against the verdicts and writes
runs/memconflict_audit30.txt. With --write-split it also writes the eval split + .meta.json.

  uv run python scripts/make_memconflict_split.py                 # audit report only
  uv run python scripts/make_memconflict_split.py --write-split   # + split files
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
from collections import Counter

from epochmem.config import repo_path
from epochmem.data.memconflict import FILTER_RULE, STRATA, MemConflictDataset, passes_quality_filter
from epochmem.data.stream import StreamSession
from epochmem.memory.extract import SYSTEM_PROMPT, render_session
from epochmem.tokens import count_text

SOURCE = {
    "repo": "https://github.com/TaoZhen1110/MemConflict",
    "revision": "ec51d5d36e87f7665d1337f3a88cbde95fc2a964",
    "file": "Data/Step4_4.jsonl",
    "url": "https://raw.githubusercontent.com/TaoZhen1110/MemConflict/ec51d5d36e87f7665d1337f3a88cbde95fc2a964/Data/Step4_4.jsonl",
    "git_blob_sha1": "6dcbf9e536ea3e5d52f015ba75b15bdcd3377c94",
    "sha256": "8ef9ec8589eccb86f63ab3a819a9180217405351a8d5846866721ea74babe092",
    "size_bytes": 39671712,
}
SPLIT_PER_STRATUM = 30
AUDIT_PER_STRATUM = 10

# verdict, one-line reason (evidence session ids in brackets are the ones read)
AUDIT_VERDICTS: dict[str, tuple[str, str]] = {
    # dynamic
    "mc-75345e85-s37-Q_004": ("clean", "S32 user: 'staying in contact'; S37 user: 'seeing people more often' -> gold from/to matches"),
    "mc-c2efaebe-s15-Q_001": ("clean", "S09 'keeping to myself' -> S15 'seeing people more often'; change = Yes"),
    "mc-a7850e51-s39-Q_003": ("clean", "S32 'Manager at IBM' -> S39 'intern at a finance startup'; Manager->Intern matches"),
    "mc-cafa273c-s09-Q_001": ("clean", "S03 'I live in New York' -> S09 'relocated to Ottawa'; Yes"),
    "mc-2e62aacf-s32-Q_002": ("clean", "S29 'I'm now single' -> S32 'now dating Ruth Anderson'"),
    "mc-8cb3c9c6-s12-Q_002": ("clean", "S01 'physically unwell and emotionally really distressed' -> S12 'physically unwell but emotionally pretty stable'"),
    "mc-f398d596-s08-Q_003": ("clean", "S06 'seeing people more often' -> S08 'spending a lot less time socializing'; Yes"),
    "mc-4cbe1c3d-s50-Q_001": ("clean", "S48 'generally healthy ... more stable' -> S50 'emotionally distressed'; Yes"),
    "mc-737e8d59-s12-Q_004": ("clean", "S07 'small tech company' -> S12 'small healthcare company'; Technology->Healthcare"),
    "mc-75b15745-s03-Q_004": ("clean", "S00 'senior at a small manufacturing company' -> S03 'startup called New Energy Tech' (name rendered as description, still matches)"),
    # static (gold = first statement Point_A by benchmark convention; later Point_B is the false contradiction)
    "mc-737e8d59-s37-Q_001": ("clean", "S01 'I was actually born in Naples'; S37 'born in Seville, Spain' (false); gold Naples"),
    "mc-8518c87c-s50-Q_001": ("clean", "S03 'studied performing arts, focusing on dance'; S50 'data scientist ... ML' (false); gold Dance"),
    "mc-f47bb65a-s17-Q_001": ("clean", "S02 'my birthday is September 15, 1983'; S17 'February 11, 1985' (false); gold 1983-09-15"),
    "mc-75b15745-s39-Q_001": ("clean", "S08 'I also have one sibling'; S39 'I have two siblings' (false); gold 1"),
    "mc-e57c356a-s23-Q_001": ("clean", "S08 'I do have siblings'; S23 'no siblings around' (weak contradiction); gold Yes"),
    "mc-f398d596-s37-Q_001": ("clean", "S04 'I'm an only child'; S37 many 'my older sister' turns (false); gold No"),
    "mc-737e8d59-s46-Q_001": ("clean", "S05 'my mom worries when I'm out late'; S46 'birth records list the mother as unknown'; gold Yes"),
    "mc-4f76dbf9-s45-Q_001": ("clean", "S07 'my dad ... born on 1963-04-12'; S45 'family records list his birth year as 1975'; gold 1963-04-12"),
    "mc-a7850e51-s36-Q_001": ("ambiguous", "S07 'my sister'; S36 'my older brother ... military' does not contradict having a sister; question asks schema field 'Sibling 1 type'"),
    "mc-e02dd733-s32-Q_001": ("clean", "S07 'my sibling was born on 1983-02-08'; S32 'her school and passport records show 1995-07-12'; gold 1983-02-08"),
    # conditional (gold = condition of the previous point's item)
    "mc-d65aae82-s25-Q_002": ("clean", "S07 'After a long workday, I find an evening walk helps me clear my head and decompress'"),
    "mc-8518c87c-s32-Q_003": ("clean", "S08 'when I'm commuting to rehearsal or need practical comfort I usually wear a techwear windbreaker'"),
    "mc-d337564c-s49-Q_002": ("clean", "S07 'I like sashimi as an occasional treat when dining out downtown or visiting the market'"),
    "mc-9dffc5de-s44-Q_003": ("clean", "S39 'I actually do coach kids' basketball sometimes at community pickup games'"),
    "mc-40a79d0c-s50-Q_009": ("clean", "S43 'the rabbit idea rests on having a quiet, stable home and getting landlord permission'"),
    "mc-2e62aacf-s25-Q_002": ("clean", "S00 'I usually rely on a waterproof raincoat while commuting in rainy weather'"),
    "mc-d65aae82-s45-Q_005": ("clean", "S25 'On a sunny weekend or a day off for a longer outdoor break, I prefer hiking or climbing'"),
    "mc-4cbe1c3d-s14-Q_001": ("clean", "S09 'adopting a very independent street cat ... provided it's suitable for a rented room and independent care'"),
    "mc-75b15745-s40-Q_002": ("clean", "S01 'haven't had a proper summer break ... daydream about a tropical island getaway for a week of real relaxation'"),
    "mc-4cbe1c3d-s52-Q_005": ("clean", "S50 'I usually clear my head by going on solo coastal hikes - they help me focus and get inspired'"),
}

NOTES = [
    "- 29/30 clean (96.7%); Wilson 95% CI for the clean rate on n=30 is roughly 83-99%, so the gate (>=85%) passes on",
    "  the point estimate but the lower bound sits just under it. In-sample there were 0 outright label errors.",
    "- Main error pattern, found OUTSIDE the random sample while reading: ENTITY MISATTRIBUTION by the dialogue",
    "  generator - a user profile value rendered as belonging to someone else. Confirmed cases: static Name items",
    "  mc-3c2e5fe5-s32-Q_001 (user's name 'Jackson Andrews' used for the child), mc-bb30a4e6-s11-Q_001, mc-e02dd733-s37-Q_001,",
    "  mc-fc1726f1-s21-Q_001 (name used for a third person), mc-2e62aacf-s27-Q_001 (name rendered as an 'old contact');",
    "  conditional mc-3c2e5fe5-s25 basketball (condition stated about the child 'Jackson', not the user).",
    "  R_name removes the 5 Name cases (5/31 static Name items); preference misattribution has no clean lexical",
    "  detector (a pronoun heuristic flagged 18 conditional items, almost all fine 'we ...' sentences).",
    "- Ambiguity pattern (#19): Point_B does not logically contradict Point_A (a brother does not refute a sister;",
    "  'records list mother as unknown' does not refute having a mom) and questions name schema fields ('Sibling 1 type').",
    "- Static gold is a CONVENTION: the first statement (Point_A) is true, the later one is the 'false contradiction';",
    "  the dialogue never marks which is false. Newer-wins precedence is therefore the WRONG policy on this stratum.",
    "- Dynamic gold is well grounded: updates are announced explicitly ('quick update: ...') in the trigger session.",
    "  38% of dynamic questions (1133/2946) are binary 'Has X changed?' -> 'Yes.' (trivially answerable without memory).",
    "- Questions are asked right after their trigger session; the history is sessions 0..trigger only (later sessions",
    "  are the question's future), so post_update == full_haystack for this dataset.",
    "- Structure: ~32 of 71k source messages are malformed (missing role/content or {'assistant': ...} keys); the loader",
    "  repairs or drops them, which leaves a few sessions with two consecutive same-role turns.",
]

# ---- candidate rule evaluated (NOT applied): lexical grounding of the gold value in the evidence ----
_STOP = set("a an and or the of to in on for with when at by is are be as from that this it its but not no yes "
            "into over after before more most less than some any".split())


def _toks(s) -> set[str]:
    s = re.sub(r"(?<=\d),(?=\d{3})", "", str(s).lower().replace("_", " "))
    return {w[:5] for w in re.findall(r"[a-z]+|\d+", s) if w not in _STOP and (len(w) >= 3 or w.isdigit())}


def _leaves(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from _leaves(x)
    elif v is not None:
        yield str(v)


def _overlap(phrase, turns) -> float:
    t = _toks(phrase)
    return max((len(t & _toks(u)) / len(t) for u in turns), default=0.0) if t else 0.0


def lexical_grounding(inst, thr: float = 0.6) -> bool:
    """Gold value tokens (>=60%) must appear in one USER turn of the relevant evidence session."""
    c, ev = inst.meta["conflict"], inst.evidence_session_ids
    if c is None or len(ev) < 2:
        return False
    by = {s.session_id: [t.content for t in s.turns if t.role == "user"] for s in inst.sessions}
    src, trg = by[ev[0]], by[ev[-1]]
    if c["kind"] == "dynamic_conflict":
        b, a = c["update"]["Before"], c["update"]["After"]
        if isinstance(a, dict) and isinstance(b, dict):
            keys = [k for k in a if b.get(k) != a.get(k) and k != "Birthdate"]
            pa, pb = [a[k] for k in keys], [b[k] for k in keys if k in b]
        else:
            pa, pb = [a], [b]
        fa = [x for v in pa for x in _leaves(v) if _toks(x)]
        fb = [x for v in pb for x in _leaves(v) if _toks(x)]
        return bool(fa) and all(_overlap(x, trg) >= thr for x in fa) and all(_overlap(x, src) >= thr for x in fb)
    sp = inst.meta["evidence_source"]["source_point"]
    if c["kind"] == "static_conflict":
        val = str(sp["Value"])
        if re.fullmatch(r"\s*(yes|no|\d+)\s*", val.lower()):
            return _overlap(sp["Target_Field_Path"].split(".")[-1].replace("Has_", ""), src) >= 0.5
        return _overlap(val, src) >= thr
    return _overlap(sp["Item"], src) >= thr and _overlap(sp["Condition"], src) >= 0.5


# ---- structure checks ----
def structure(inst) -> dict:
    dates = [s.date for s in inst.sessions]
    alt_bad = sum(any(a.role == b.role for a, b in zip(s.turns, s.turns[1:])) for s in inst.sessions)
    return {
        "chronological": dates == sorted(dates) and inst.question_date >= dates[-1],
        "empty_sessions": sum(not s.turns for s in inst.sessions),
        "first_turn_user": sum(bool(s.turns) and s.turns[0].role == "user" for s in inst.sessions),
        "sessions_with_role_repeat": alt_bad,
        "n_sessions": len(inst.sessions),
    }


def audit_sample(insts, seed: int) -> list:
    out = []
    for st in STRATA:
        pool = sorted((i for i in insts if i.stratum == st), key=lambda i: i.qid)
        out.extend(random.Random(f"{seed}:audit:{st}").sample(pool, AUDIT_PER_STRATUM))
    return out


def split_sample(insts, seed: int) -> list:
    out = []
    for st in STRATA:
        pool = sorted((i for i in insts if i.stratum == st and passes_quality_filter(i)), key=lambda i: i.qid)
        out.extend(random.Random(f"{seed}:eval90:{st}").sample(pool, SPLIT_PER_STRATUM))
    return out


def history_stats(inst, sys_tokens: int) -> dict:
    turns = sum(len(s.turns) for s in inst.sessions)
    raw = sum(count_text(t.content) for s in inst.sessions for t in s.turns)
    ext = [sys_tokens + count_text(render_session(StreamSession(k + 1, s, False), 2000, 300))
           for k, s in enumerate(inst.sessions)]
    return {"sessions": len(inst.sessions), "turns": turns, "raw_tokens": raw, "extract_in_tokens": sum(ext),
            "extract_in_per_session": ext}


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else "n/a"


def write_audit(insts, sample, path, seed: int) -> dict:
    missing = [i.qid for i in sample if i.qid not in AUDIT_VERDICTS]
    if missing:
        raise SystemExit(f"audit sample changed; no verdict for {missing}")
    lines = [
        "MemConflict Step4_4.jsonl - 30-question label audit (read by an LLM assistant; no automated judge, no model outputs)",
        f"source: {SOURCE['repo']} @ {SOURCE['revision']} {SOURCE['file']} sha256={SOURCE['sha256']}",
        f"sample: seed={seed}, random.Random('{seed}:audit:<stratum>').sample(pool sorted by qid, 10) per conflict type",
        "check per item: dialogue structure, session ordering/dates, and whether the gold answer is stated (for the USER)",
        "in the evidence sessions (gold answers are generated from construction metadata, not from the dialogue).",
        "",
        f"{'#':>2}  {'qid':<24} {'type':<12} {'verdict':<10} {'R_name':<6} {'R_lex':<5} reason",
    ]
    by_type: dict[str, Counter] = {}
    rows = []
    for n, i in enumerate(sample, 1):
        v, why = AUDIT_VERDICTS[i.qid]
        st = structure(i)
        rn, rl = passes_quality_filter(i), lexical_grounding(i)
        rows.append((i, v, rn, rl, st))
        by_type.setdefault(i.qtype, Counter())[v] += 1
        lines.append(f"{n:>2}  {i.qid:<24} {i.qtype.split('-')[1]:<12} {v:<10} {'keep' if rn else 'drop':<6} "
                     f"{'keep' if rl else 'drop':<5} {why}")
        lines.append(f"    Q: {i.question}  | gold: {i.answer[:140]}")
        lines.append(f"    evidence={list(i.evidence_session_ids)} sessions={st['n_sessions']} chronological={st['chronological']} "
                     f"empty_sessions={st['empty_sessions']} sessions_with_role_repeat={st['sessions_with_role_repeat']}")
    clean = sum(v == "clean" for _, v, *_ in rows)
    lines += ["", "== clean-label rate ==", f"overall: {_pct(clean, len(rows))}"]
    for t, c in by_type.items():
        lines.append(f"{t}: {_pct(c['clean'], sum(c.values()))}  ({dict(c)})")
    lines += ["", "== structure (all 30) ==",
              f"chronological sessions + question_date >= last session date: {sum(r[4]['chronological'] for r in rows)}/30",
              f"empty sessions: {sum(r[4]['empty_sessions'] for r in rows)}; sessions whose first turn is the user: "
              f"{sum(r[4]['first_turn_user'] for r in rows)}/{sum(r[4]['n_sessions'] for r in rows)}",
              f"sessions with a repeated role (non-alternating, from malformed source messages): "
              f"{sum(r[4]['sessions_with_role_repeat'] for r in rows)}"]

    def rule_line(name, idx):
        kept = [r for r in rows if r[idx]]
        dropped = [r for r in rows if not r[idx]]
        kc = sum(r[1] == "clean" for r in kept)
        dbad = sum(r[1] != "clean" for r in dropped)
        return (f"{name}: keeps {len(kept)}/30, clean among kept {_pct(kc, len(kept))}; drops {len(dropped)} "
                f"(of which non-clean {dbad}) -> precision of 'drop' as error detector {_pct(dbad, len(dropped))}")

    full = Counter((i.qtype, passes_quality_filter(i)) for i in insts)
    lex = Counter((i.qtype, lexical_grounding(i)) for i in insts)
    lines += ["", "== filter rules (evaluated on the 30) ==",
              f"R_name (APPLIED): {FILTER_RULE}",
              "  " + rule_line("R_name", 2),
              "  full dataset: drops " + ", ".join(f"{t}={full[(t, False)]}" for t in STRATA)
              + " (all 5 dropped Name items were read: user's name rendered as a third party / non-self name)",
              "R_lex (NOT applied): gold value tokens (>=60%, condition >=50%) must occur in one user turn of the evidence session",
              "  " + rule_line("R_lex", 3),
              "  full dataset: drops " + ", ".join(f"{t}={lex[(t, False)]}/{lex[(t, False)] + lex[(t, True)]}" for t in STRATA)
              + "; spot-checked drops were paraphrases ('I'm a woman', 'my mom', 'constantly tired'), i.e. false alarms",
              ]
    lines += ["", "== notes / error patterns ==", *NOTES]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {"clean": clean, "n": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/raw/memconflict/Step4_4.jsonl")
    ap.add_argument("--audit-out", default="runs/memconflict_audit30.txt")
    ap.add_argument("--out", default="data/raw/memconflict/memconflict_eval90.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--write-split", action="store_true")
    args = ap.parse_args()

    src = repo_path(args.source)
    ds = MemConflictDataset(src, expected_sha256=SOURCE["sha256"])
    insts = ds.load()
    print(f"loaded {len(insts)} questions: {dict(Counter(i.qtype for i in insts))}")
    res = write_audit(insts, audit_sample(insts, args.seed), repo_path(args.audit_out), args.seed)
    rate = res["clean"] / res["n"]
    print(f"audit: clean {res['clean']}/{res['n']} = {rate:.1%} -> {repo_path(args.audit_out)}")
    if not args.write_split:
        return 0
    if rate < 0.85:
        raise SystemExit("gate failed: clean-label rate < 85%, split not written")

    sample = split_sample(insts, args.seed)
    qids = [i.qid for i in sample]
    need: dict[str, int] = {}
    for i in sample:
        need[i.meta["user_id"]] = max(need.get(i.meta["user_id"], -1), i.meta["trigger_session"])
    out = repo_path(args.out)
    lines = []
    for u in ds.users():                     # verbatim user records, truncated after the last needed session
        if u["ID"] in need:
            u = dict(u, Full_Session_Chain=u["Full_Session_Chain"][: need[u["ID"]] + 1])
            lines.append(json.dumps(u, ensure_ascii=False))
    blob = "\n".join(lines) + "\n"
    out.write_text(blob, encoding="utf-8", newline="\n")
    check = MemConflictDataset(out, qids=qids).load()
    assert [i.qid for i in check] == qids and all(a.sessions == b.sessions for a, b in zip(check, sample))

    sys_tokens = count_text(SYSTEM_PROMPT.format(max_facts=10)) + 40   # + EXISTING KEYS header/registry floor
    stats = {i.qid: history_stats(i, sys_tokens) for i in sample}
    uniq = {}
    for i in sample:
        for k, s in enumerate(i.sessions):
            uniq[(i.meta["user_id"], k)] = stats[i.qid]["extract_in_per_session"][k]

    def summ(key):
        v = [s[key] for s in stats.values()]
        return {"mean": round(statistics.mean(v), 1), "median": statistics.median(v), "min": min(v), "max": max(v),
                "sum": sum(v)}

    per_stratum = {st: {k: round(statistics.mean(stats[i.qid][k] for i in sample if i.qtype == st), 1)
                        for k in ("sessions", "turns", "raw_tokens", "extract_in_tokens")} for st in STRATA}
    meta = {
        "dataset": "memconflict", "source": SOURCE, "n": len(qids), "qids": qids, "seed": args.seed,
        "selection": f"per stratum random.Random('{args.seed}:eval90:<stratum>').sample(sorted filtered pool, "
                     f"{SPLIT_PER_STRATUM}); strata {list(STRATA)}",
        "filter_rule": FILTER_RULE,
        "filter_dropped_from_pool": {st: sum(1 for i in insts if i.qtype == st and not passes_quality_filter(i)) for st in STRATA},
        "audit": {"file": args.audit_out, "clean": res["clean"], "n": res["n"]},
        "split_file_note": "verbatim Step4_4 user records for the users referenced, each truncated after the last "
                           "trigger session needed; load with MemConflictDataset(path, qids=meta['qids'])",
        "split_sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "history": {"sessions": summ("sessions"), "turns": summ("turns"), "raw_tokens": summ("raw_tokens"),
                    "extract_in_tokens": summ("extract_in_tokens"), "per_stratum_mean": per_stratum,
                    "unique_user_sessions": len(uniq), "unique_extract_in_tokens": sum(uniq.values()),
                    "note": "extract_in_tokens = o200k tokens of the extractor prompt (system + render_session with "
                            "user 2000 / assistant 300 char truncation) summed over the question's sessions; "
                            "unique_* counts each (user, session) once, which is what a chronological, cached "
                            "extractor pays because questions of one user share the session prefix"},
        "binary_yes_answers": sum(1 for i in sample if re.match(r"^\s*yes\b", i.answer, re.I)),
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(f"wrote {len(qids)} qids from {len(need)} users -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(meta["history"], indent=1))
    print("binary 'Yes' gold answers in split:", meta["binary_yes_answers"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
