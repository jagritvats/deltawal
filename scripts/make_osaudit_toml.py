"""Generate osaudit.toml: trace every number printed in paper/main.pdf to results/paper_numbers.json.

osaudit (https://github.com/ASSERT-KTH/osaudit) reads the numbers of the PDF with poppler's `pdftotext -layout`
(the prose numbers are \\nm{...} lookups, so the LaTeX source holds no digits to audit). For each distinct
printed number this script writes

  [[numbers.derive]]  naming the results value it prints (preferring the manifest key the prose looks up with
                      \\nm, then the generated table cell, then any other value; `a - b` for \\nmdiff), or
  [[numbers.allow]]   for a number the study does not produce (a citation year, a section number, an axis
                      tick), scoped to its page and occurrence, with the reason of its ALLOW rule.

A derivation covers every occurrence of its printed value, so a digit of a model name (glm-5.3) that happens to
equal a results value is covered by that value's derivation.

A printed number that is neither traced nor covered by an ALLOW rule stops the script (exit 1), so a new
untraced number is never allowed silently. Needs osaudit (Python >= 3.13) and poppler-utils:

  osaudit run --input paper/data/snapshot.json.gz --input configs/m2_heldout.yaml \\
      -o results/paper_numbers.json -- python scripts/emit_paper_numbers.py
  python scripts/make_osaudit_toml.py                 # writes osaudit.toml
  osaudit                                             # prints nothing, exit 0

  --text FILE   use this `pdftotext -layout paper/main.pdf -` output instead of running pdftotext
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from osaudit import _pdf
from osaudit._config import AUDIT
from osaudit._number import Num, close, make
from osaudit._results import Results, _name

ROOT = Path(__file__).resolve().parents[1]
PDF = "paper/main.pdf"
RESULTS = "results/paper_numbers.json"
STEM = Path(RESULTS).stem
INSTRUMENTS = ["scripts/emit_paper_numbers.py", "scripts/build_paper_assets.py"]
SECTIONS = sorted((ROOT / "paper" / "sections").glob("*.tex"))
_KEY_BAD = re.compile(r"[^A-Za-z0-9_\-.]")         # = build_paper_assets._KEY_BAD (keys of paper/numbers.tex)

# Printed numbers the study does not produce: (regex on the number as printed, regex on its pdftotext line in
# which {n} stands for the number, reason). An occurrence is allowed by the first rule whose number regex matches
# it and whose line regex matches a span of its line that contains it.
ALLOW: list[tuple[str, str, str]] = [
    (r"19[7-9]\d|20[0-2]\d", r"(?:\b(?:al\.|Lamport|Sinchaisri),?\s*\(?|;\s*|^){n}[);]",
     "publication year of a cited work (author-year citation)"),
    (r"\d\.\d", r"^{n}\s{2,}[A-Z]", "section number of a heading"),
    (r"\d\.\d", r"tion {n}\)", "section number of a cross-reference hyphenated across lines (Sec-tion)"),
    (r"10[0-4]", r"^(?:10[0-4]\s+)*{n}(?:\s+10[0-4])*$",
     "tick 10^k of a log-scale latency axis in Figure 2 (pdftotext flattens the superscript)"),
    # today covered by derivations of equal results values; kept so a results change cannot make them untraced
    (r"\d+(?:\.\d+)?", r"\b(?:glm|gpt|claude-haiku)-(?:\d+[.-])*{n}", "digits of a model identifier"),
]


def pdf_text(args: argparse.Namespace) -> str:
    if args.text:
        return Path(args.text).read_text(encoding="utf-8")
    exe = shutil.which("pdftotext")
    if exe is None:
        sys.exit("pdftotext not found: install poppler-utils (osaudit reads the PDF with it)")
    version = subprocess.run([exe, "-v"], capture_output=True, text=True)
    if "poppler" not in (version.stdout + version.stderr).lower():
        sys.exit(f"{exe} is not poppler's pdftotext (osaudit uses poppler; xpdf extracts differently)")
    return _pdf.text_of(ROOT / PDF)


def occurrences(text: str) -> list[tuple[Num, str, int]]:
    """Every number osaudit audits in the PDF, with the text line it sits on and its column there.

    Masks exactly as osaudit._pdf.numbers does, but blanks a masked span instead of collapsing it, so the
    positions (and the line of each number) survive. Checked against osaudit's own list below.
    """
    out = []
    for page, body in enumerate(_pdf.pages(text), 1):
        masked = body
        for pat in (_pdf.QUOTE, _pdf.CITATION, _pdf.PART, _pdf.LONE_NUMBER, _pdf.LIST_LABEL):
            masked = pat.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), masked)
        for m in _pdf.PDF_NUMBER.finditer(masked):
            start = body.rfind("\n", 0, m.start()) + 1
            end = body.find("\n", m.end())
            line = body[start:end if end >= 0 else len(body)]
            out.append((make(PDF, page, m.group(0).strip(), m.group(1), m.group(2), m.group(3) is not None),
                        line, m.start() - start))
    reference = [(n.line, n.key) for n in _pdf.numbers(PDF, text)]
    if [(n.line, n.key) for n, _, _ in out] != reference:
        sys.exit("number inventory differs from osaudit's (a masked span crosses a line); update occurrences()")
    return out


def manifest_names(results: Results) -> dict[str, str]:
    """paper/numbers.tex key (as \\nm{...} names it) -> results value name, for the `metrics` block."""
    out: dict[str, str] = {}

    def walk(name: str, key: str, v) -> None:
        if isinstance(v, dict):
            if {"mean", "lo", "hi"} <= set(v):      # a CI statistic: the bare key is its point estimate
                out[key] = _name(name, "mean")
            for k, x in v.items():
                walk(_name(name, k), f"{key}/{k}" if key else str(k), x)
        elif isinstance(v, list):
            for i, x in enumerate(v):
                walk(_name(name, i), f"{key}/{i}", x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out["/".join(_KEY_BAD.sub("_", s).strip("_") for s in key.split("/"))] = name

    walk(_name(STEM, "metrics"), "", results.docs[STEM]["metrics"])
    return out


def rounded_names(results: Results) -> dict[str, str]:
    """paper/numbers.tex key -> results value name, for the `rounded` block (values \\nm prints as integers)."""
    block = _name(STEM, "rounded")
    return {"/".join(_KEY_BAD.sub("_", s).strip("_") for s in key.split("/")): _name(block, key)
            for key in results.docs[STEM]["rounded"]}


def prose_keys() -> tuple[set[str], list[tuple[str, str]]]:
    """The manifest keys the prose prints (\\nm, \\nmci), and the \\nmdiff pairs."""
    keys, diffs = set(), []
    for f in SECTIONS:
        tex = f.read_text(encoding="utf-8")
        keys.update(re.findall(r"\\nm\{([^{}]*)\}", tex))
        for k in re.findall(r"\\nmci\{([^{}]*)\}", tex):
            keys.update((f"{k}/lo", f"{k}/hi"))
        diffs += re.findall(r"\\nmdiff\{([^{}]*)\}\{([^{}]*)\}", tex)
    return keys, diffs


def allowed(text: str, line: str, col: int, value: str, pattern: str) -> bool:
    """Whether the rule (value, pattern) covers the number `text` printed at column `col` of `line`."""
    if not re.fullmatch(value, text):
        return False
    rx = re.compile(pattern.replace("{n}", re.escape(text)))
    col -= len(line) - len(line.lstrip())                # ^ and $ are the ends of the line's text
    return any(m.start() <= col < m.end() for m in rx.finditer(line.strip()))


def toml_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", help="pdftotext -layout output of paper/main.pdf")
    ap.add_argument("-o", "--output", default=str(ROOT / "osaudit.toml"))
    args = ap.parse_args()

    occ = occurrences(pdf_text(args))
    results = Results.load([ROOT / RESULTS])
    values = {k: v for k, v in results.values.items() if not k.endswith(" (key)")}
    by_key = manifest_names(results)
    used, diffs = prose_keys()
    rounded = rounded_names(results)
    tier0 = {by_key[k] for k in used if k in by_key} | {rounded[k] for k in used if k in rounded}
    tables = _name(STEM, "tables")

    def rank(name: str) -> tuple:
        return (0 if name in tier0 else 1 if name.startswith(tables) else 2, len(name), name)

    derive: dict[tuple, tuple[str, str]] = {}        # num.key -> (printed value, expr)
    for n, _, _ in occ:
        if n.key in derive:
            continue
        names = sorted((k for k, v in values.items() if close(n, v)), key=rank)
        if names:
            derive[n.key] = (n.text, names[0])
            continue
        for a, b in diffs:
            if a in by_key and b in by_key:
                expr = f"{by_key[a]} - {by_key[b]}"
                if close(n, results.evaluate(expr)):
                    derive[n.key] = (n.text, expr)
                    break
        else:
            negated = sorted((k for k, v in values.items() if v < 0 and close(n, -v)), key=rank)
            if negated:
                derive[n.key] = (n.text, f"-{negated[0]}")

    allow: list[tuple[Num, str, int]] = []           # (number, reason, occurrence on its page)
    seen: dict[tuple, int] = defaultdict(int)
    missing = []
    for n, line, col in occ:
        seen[(n.line, n.key)] += 1
        if n.key in derive:
            continue
        reason = next((r for value, pat, r in ALLOW if allowed(n.text, line, col, value, pat)), None)
        if reason is None:
            missing.append(f"{PDF}:{n.line}: {n.text!r} untraced in: {line.strip()}")
        else:
            allow.append((n, reason, seen[(n.line, n.key)]))
    if missing:
        print("\n".join(missing), file=sys.stderr)
        print(f"{len(missing)} printed number(s) neither traced nor allowed: add a results value or an ALLOW rule",
              file=sys.stderr)
        return 1

    out = [f"schema = {toml_str(AUDIT)}",
           "# generated by scripts/make_osaudit_toml.py -- do not edit by hand; regenerate after rebuilding the PDF",
           "",
           "[paper]",
           f"pdf = {toml_str(PDF)}            # the prose numbers are \\nm{{}} lookups: only the PDF prints them",
           "",
           "[results]",
           f"files = [{toml_str(RESULTS)}]",
           f"instruments = [{', '.join(map(toml_str, INSTRUMENTS))}]",
           "",
           "[[results.depends]]",
           f"file = {toml_str(RESULTS)}",
           f"instruments = [{', '.join(map(toml_str, INSTRUMENTS))}]",
           ""]
    for key, (text, expr) in sorted(derive.items(), key=lambda kv: kv[0]):
        out += ["[[numbers.derive]]", f"value = {toml_str(text)}", f"expr = {toml_str(expr)}", ""]
    for n, reason, k in sorted(allow, key=lambda a: (a[0].line, a[0].key, a[2])):
        out += ["[[numbers.allow]]", f"value = {toml_str(n.text.replace(' ', ''))}", f"reason = {toml_str(reason)}",
                f"file = {toml_str(PDF)}", f"line = {n.line}                      # page", f"occurrence = {k}", ""]
    Path(args.output).write_text("\n".join(out), encoding="ascii", newline="\n")
    print(f"{args.output}: {len(derive)} derive, {len(allow)} allow entries "
          f"for {len(occ)} printed numbers ({len({o[0].key for o in occ})} distinct)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
