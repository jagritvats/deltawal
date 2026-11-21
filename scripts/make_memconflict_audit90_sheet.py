"""Human-review sheet for the full audit of the 90 evaluated MemConflict questions.

Blind by construction: reads ONLY the raw MemConflict split (questions, gold answers, conflict records,
dialogues) and the frozen audit verdicts in runs/memconflict_audit90.jsonl. It never opens model
predictions or judge verdicts.

For each question it locates every quoted evidence snippet in the named session and prints the FULL
dialogue turn that contains it (so the reviewer sees the exact line, not a paraphrase).

Writes runs/memconflict_audit90.md (read) and runs/memconflict_audit90.csv (one row per question, with an
empty `human_verdict` / `human_note` column to fill in).

  uv run python scripts/make_memconflict_audit90_sheet.py
"""
from __future__ import annotations

import csv
import json
import re

from epochmem.config import load_config, repo_path
from epochmem.runner import select_instances

VERDICTS = repo_path("runs/memconflict_audit90.jsonl")
VERDICTS_R2 = repo_path("runs/memconflict_audit90_r2.jsonl")   # independent second rater (other model)
TYPE_ORDER = {"dynamic": 0, "static": 1, "conditional": 2}

# Rubric given verbatim to both blind raters (rendered into the paper's prompt appendix).
AUDIT_RUBRIC = """Conflict types:
- dynamic: a user attribute changes over time; earlier session = old value, later session = new value; gold describes the change or the current value ("Has X changed?" -> "Yes").
- static: by the dataset's convention the FIRST statement is true and a LATER statement is a false contradiction of it; gold = the first value.
- conditional: gold = the context/condition under which the user prefers something; the user must state it about themselves.

Give each question exactly one verdict:
- clean: the user's own statements support the gold as worded, the value is about the user, and the question is answerable as worded. For static, the later statement must genuinely contradict the earlier one.
- ambiguous: the relevant statements exist, but a careful reader could defend a different answer, or the conflict the item is built on is not a real contradiction/change, or a key statement is made only by the assistant.
- label_error: the gold is unsupported or contradicted by the dialogue, the needed statement is missing, or the value belongs to someone other than the user.

Method: from GOLD and CONFLICT RECORD pick key terms, Grep the file for them and paraphrases, Read the surrounding lines. Decide from quoted dialogue, not from the gold. Be consistent."""


_TYPO = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(_TYPO)).strip().lower()


def main() -> int:
    verdicts = {r["qid"]: r for r in (json.loads(l) for l in VERDICTS.open(encoding="utf-8") if l.strip())}
    r2 = ({r["qid"]: r for r in (json.loads(l) for l in VERDICTS_R2.open(encoding="utf-8") if l.strip())}
          if VERDICTS_R2.exists() else {})
    cfg = load_config("configs/m3_memconflict.yaml")
    insts = {p.inst.qid: p.inst for p in select_instances(cfg, None)}
    assert set(insts) == set(verdicts), (len(insts), len(verdicts))
    def status(q):
        a = verdicts[q]["verdict"] == "clean"
        b = (r2[q]["verdict"] == "clean") if q in r2 else a
        return 0 if a != b else (1 if not a else 2)          # disagreement, flagged by both, clean by both
    order = sorted(insts, key=lambda q: (status(q), TYPE_ORDER[insts[q].qtype.split("-")[1]], q))
    md = ["# MemConflict full audit: all 90 evaluated questions", "",
          "Blind audit: verdicts were assigned from the dialogues, questions and gold answers only (no model outputs).",
          "Who: AI reading agents (Claude), one pass per question, every verdict backed by verbatim quotes; the 14",
          "flagged items were re-read once more by the coordinating assistant. A second, independent rater pass with a",
          "different model (Sonnet; same files, same blindness, never saw round-1 verdicts) is shown per question.",
          "Order: rater disagreements first, then flagged by both, then clean by both.",
          "NOT yet reviewed by a human: use the human_verdict / human_note columns of the .csv.",
          "Flagged = ambiguous or label_error. Evidence lines are the full dialogue turns containing each quoted snippet.",
          ""]
    n = {"clean": 0, "ambiguous": 0, "label_error": 0}
    for q in order:
        n[verdicts[q]["verdict"]] += 1
    md.append("| verdict | count |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in n.items()))
    md.append("")
    rows, missing = [], 0
    for q in order:
        inst, v = insts[q], verdicts[q]
        t = inst.qtype.split("-")[1]
        sess = {s.session_id: s for s in inst.sessions}
        lines = []
        for quote in v.get("quotes", []):
            sid = quote["session"]
            hit = None
            for s_id in ([sid] if sid in sess else []) + [x for x in inst.evidence_session_ids if x != sid]:
                for tr in sess[s_id].turns:
                    if _norm(quote["text"]) in _norm(tr.content):
                        hit = (s_id, sess[s_id].date, tr.role, tr.content.replace("\n", " "))
                        break
                if hit:
                    break
            if hit:
                lines.append(f"{hit[0]} ({hit[1]}) [{hit[2]}] {hit[3]}")
            else:
                missing += 1
                lines.append(f"{sid} [QUOTE NOT FOUND VERBATIM] {quote['text']}")
        flag = "clean" if v["verdict"] == "clean" else "flagged"
        v2 = r2.get(q)
        tag = {0: "RATERS DISAGREE", 1: "flagged by both", 2: "clean by both"}[status(q)] if v2 else flag
        md += [f"## {q} — {t} — **{tag}** (rater 1: {v['verdict']}" + (f"; rater 2: {v2['verdict']})" if v2 else ")"),
               f"- **Question:** {inst.question}", f"- **Gold:** {inst.answer}",
               f"- **Evidence sessions:** {', '.join(inst.evidence_session_ids)}",
               f"- **Rater 1 reason:** {v['reason']}"] + ([f"- **Rater 2 reason:** {v2['reason']}"] if v2 else []) + [
               "- **Evidence lines (rater 1 quotes):**"]
        md += [f"  - {l}" for l in lines]
        md.append("")
        rows.append({"qid": q, "conflict_type": t, "status": tag, "audit": flag, "audit_verdict": v["verdict"],
                     "reason": v["reason"], "rater2_verdict": (v2 or {}).get("verdict", ""),
                     "rater2_reason": (v2 or {}).get("reason", ""),
                     "question": inst.question, "gold": inst.answer,
                     "evidence_lines": " || ".join(lines), "human_verdict": "", "human_note": ""})
    out_md, out_csv = repo_path("runs/memconflict_audit90.md"), repo_path("runs/memconflict_audit90.csv")
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} questions {n}; quotes not found verbatim: {missing}")
    print(f"wrote {out_md} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
