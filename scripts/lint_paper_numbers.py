"""Flag numerals typed by hand in the paper prose (paper/main.tex + paper/sections/*.tex).

Every quantitative number must come from paper/numbers.tex (\\nm, \\nmci, \\nmdiff, \\nmraw). After stripping
comments, those macros, references/citations/labels/inputs/graphics, and an allow-list of names that contain
digits (model ids, dataset names, version families) and design-setting labels (lag values, p_ooo), any
remaining digit is reported with file:line. Exit 1 if anything is found.

  uv run python scripts/lint_paper_numbers.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "paper" / "main.tex", *sorted((ROOT / "paper" / "sections").glob("*.tex"))]

STRIP = [
    r"(?<!\\)%.*$",                                                     # comments
    r"\\nm(ci|raw)?\{[^{}]*\}", r"\\nmdiff\{[^{}]*\}\{[^{}]*\}",
    r"\\(c|C)?ref\{[^{}]*\}", r"\\(auto|page|eq)?ref\{[^{}]*\}", r"\\label\{[^{}]*\}",
    r"\\cite[pt]?\*?(\[[^\]]*\])*\{[^{}]*\}", r"\\input\{[^{}]*\}", r"\\includegraphics(\[[^\]]*\])?\{[^{}]*\}",
    r"\\(url|href)\{[^{}]*\}", r"\\(begin|end)\{[^{}]*\}(\{[^{}]*\})?(\[[^\]]*\])?",
    r"\\(v|h)space\*?\{[^{}]*\}", r"\[[a-z!]*\]",                       # float placement
    r"width=[0-9.]*\\(textwidth|linewidth|columnwidth)",
]
ALLOW = [
    r"gemma4(:12b-it-qat)?", r"\b12B\b", r"glm-5\.3(-flash)?", r"gpt-6-luna", r"qwen3\.8-flash", r"qwen4-max",
    r"qwen3-embedding(:0\.6b)?", r"Mem0\s*(OSS)?\s*\\?nm\{[^}]*\}", r"Mem0\s+2\.x", r"\bmem0ai\b",
    r"LongMemEval(-S)?", r"\bB[123]\b", r"\bB[123]_[a-z_]+", r"\bk\b",
    r"\blags?~?\s*\$?\{?[0-5](\s*(,|/|and|or|--|\\text\{--\}|\\in\\\{)\s*[0-5])*\}?\$?",
    r"\blag-[0-5]\b", r"\$?\\?lag\s*=\s*[0-5]\$?",r"\$lag\s*=\s*[0-5]\$", r"\\ell\s*=\s*[0-5]", r"\\ell\s*\\geq\s*1",
    r"p_\{?\\text\{ooo\}\}?\s*(=|\\in)\s*\\?\{?\s*0(\.0)?(\s*,\s*0\.3)?\s*\\?\}?", r"p_\{?\\text\{ooo\}\}?\s*=\s*0\.3",
    r"\bsession(s)?\s+\$?1\s*(\.\.|\\ldots|--)\s*T(\s*-\s*1)?\$?", r"\$?T\s*-\s*1\$?", r"\$1\.\.T\$",
    r"\bstep[s]?\s+[1-9]\b", r"\(\s*[ivx]+\s*\)", r"\bO\(1\)", r"\btop-\$?k\$?", r"\b95th percentile\b",
    r"\\text\{[^{}]*\}",
]


def scrub(line: str) -> str:
    for rx in STRIP:
        line = re.sub(rx, " ", line)
    for rx in ALLOW:
        line = re.sub(rx, " ", line)
    return line


def main() -> int:
    bad = []
    for f in FILES:
        if not f.exists():
            continue
        in_verbatim = False
        for i, raw in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if raw.lstrip().startswith(("\\usepackage", "\\documentclass", "\\def\\month", "\\def\\year",
                                        "\\def\\openreview", "\\bibliography", "\\setlength")):
                continue
            s = scrub(raw)
            if re.search(r"\d", s):
                bad.append(f"{f.relative_to(ROOT).as_posix()}:{i}: {raw.strip()[:160]}")
    for b in bad:
        print(b)
    print(f"{len(bad)} line(s) with hand-typed numerals" if bad else "no hand-typed numerals")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
