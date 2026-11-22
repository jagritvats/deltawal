"""Print, as one JSON object on stdout, every number the paper's generated assets print.

The paper never types a number by hand: prose numbers come from paper/numbers.tex (\\nm lookups into the
numbers manifest) and table cells from paper/tables/*.tex, both written by scripts/build_paper_assets.py from
the committed snapshot paper/data/snapshot.json.gz. This instrument reruns that build from the snapshot in a
temporary directory (the committed assets are not touched) and prints

  metrics   the numbers manifest (`metrics` of paper/numbers_manifest.json), unchanged
  rounded   manifest values that \\nm prints as integers (1256.5 tokens -> 1256), as printed
  tables    per generated table: the numbers of its caption and of each body row, as printed (same rounding)

so that `osaudit` can trace every printed number to a results value (see osaudit.toml). It is run through
`osaudit run`, which records the provenance block in results/paper_numbers.json:

  osaudit run --input paper/data/snapshot.json.gz --input configs/m2_heldout.yaml \\
      -o results/paper_numbers.json -- python scripts/emit_paper_numbers.py

with a `python` that has the project and paper dependencies (as in `uv run --group paper`); then
scripts/make_osaudit_toml.py writes osaudit.toml.

Deterministic: the build is byte-identical across runs, and nothing here reads the clock.
"""
from __future__ import annotations

import contextlib
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]     # this checkout's code, not an installed copy

import build_paper_assets as bpa  # noqa: E402

# A number as osaudit reads one: not glued to a word, a path or a command (`12B`, `RQ1`, `3.0cm`).
NUMBER = re.compile(r"(?<![\w.\\/])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![.,]?\w)")
# LaTeX that holds digits which are not printed numbers.
NOT_PRINTED = [
    r"(?<!\\)%.*$",                                                   # comments
    r"\\begin\{tabular\*?\}(\{[^{}]*\})?\{(?:[^{}]|\{[^{}]*\})*\}",   # column specs
    r"\\multicolumn\{\d+\}\{[^{}]*\}", r"\\cmidrule(\([a-z]*\))?\{[^{}]*\}",
    r"\\(set|add)length\{[^{}]*\}\{[^{}]*\}", r"\\label\{[^{}]*\}", r"\\(c|C)?ref\{[^{}]*\}",
    r"\\arraystretch\}\{[^{}]*\}", r"\\(hspace|vspace)\*?\{[^{}]*\}", r"\\rule(\[[^\]]*\])?\{[^{}]*\}\{[^{}]*\}",
    r"\\[a-zA-Z]+",                                                   # command names (\tabcolsep, \small)
]


def _number(integer: str, frac: str | None) -> int | float:
    digits = integer.replace(",", "")
    return float(digits + frac) if frac else int(digits)


def numbers_in(tex: str) -> list[int | float]:
    tex = tex.replace("{,}", ",")
    for rx in NOT_PRINTED:
        tex = re.sub(rx, " ", tex, flags=re.MULTILINE)
    return [_number(m.group(1), m.group(2)) for m in NUMBER.finditer(tex)]


def caption_of(tex: str) -> str:
    """The argument of \\caption{...}, braces balanced."""
    i = tex.find("\\caption{")
    if i < 0:
        return ""
    depth, j = 0, i + len("\\caption")
    for k in range(j, len(tex)):
        depth += {"{": 1, "}": -1}.get(tex[k], 0)
        if depth == 0:
            return tex[j + 1:k]
    return ""


def table(tex: str) -> dict:
    """The printed numbers of one generated table: its caption, then each body row."""
    caption = caption_of(tex)
    body = tex.replace(caption, " ") if caption else tex
    rows = []
    for line in body.splitlines():
        line = re.sub(r"(?<!\\)%.*$", "", line)
        if "&" not in line:
            continue
        cells = line.split("&")
        values = numbers_in("&".join(cells[1:]))
        label_values = numbers_in(cells[0])
        if values or label_values:
            label = re.sub(r"\\[a-zA-Z]+|[{}$\\]", "", cells[0]).strip()
            rows.append({"label": label, "values": label_values + values})
    return {"caption": numbers_in(caption), "rows": rows}


def rounded(metrics: dict) -> dict[str, int]:
    """Manifest values that paper/numbers.tex prints as integers although they are not (1256.5 -> 1256), keyed
    by manifest key: the printed integer, rounded exactly as \\nm{key} prints it (build_paper_assets._fmt_value)."""
    flat: dict = {}
    for k, v in metrics.items():
        bpa._flatten(k, v, flat)
    out = {}
    for key, v in sorted(flat.items()):
        if isinstance(v, float) and v != int(v):
            shown = bpa._fmt_value(key, v).replace("{,}", "")
            if re.fullmatch(r"\d+", shown):
                out[key] = int(shown)
    return out


def main() -> int:
    snap = bpa.read_snapshot()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(sys.stderr):
        bpa.PAPER = Path(tmp)                       # the build writes its assets here, not into paper/
        manifest = bpa.build(snap)
        tables = {p.stem: table(p.read_text(encoding="utf-8"))
                  for p in sorted((Path(tmp) / "tables").glob("tab*.tex"))}
    print(json.dumps({"metrics": manifest["metrics"], "rounded": rounded(manifest["metrics"]), "tables": tables},
                     indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
