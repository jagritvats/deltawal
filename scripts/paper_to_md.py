"""Markdown twin of the compiled paper (paper/main.md), for internal reading and as context for tools.

Same sources as the PDF: paper/main.tex + sections + generated tables. Every \\nm / \\nmci / \\nmdiff
macro is expanded from paper/numbers.tex (the values the PDF prints), \\ref labels are resolved from the
LaTeX run's paper/main.aux (so Table/Figure/Section numbers match the PDF), generated tables become pipe
tables, figures point at paper/figures/preview/*.png, and citations are rendered by pandoc's citeproc from
paper/references.bib. Run `bash paper/compile_paper.sh` first (it writes main.aux).

  uv run --with pypandoc_binary python scripts/paper_to_md.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"


def load_numbers() -> tuple[dict, dict]:
    txt = (P / "numbers.tex").read_text(encoding="utf-8")
    fmt = dict(re.findall(r"\\csname nm@(.+?)\\endcsname\{(.*)\}$", txt, re.M))
    raw = {k: float(v) for k, v in re.findall(r"\\csname nmraw@(.+?)\\endcsname\{(.*)\}$", txt, re.M)}
    return fmt, raw


def load_labels() -> dict:
    aux = (P / "main.aux").read_text(encoding="utf-8")
    return dict(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux))


def expand_numbers(s: str, fmt: dict, raw: dict) -> str:
    def nm(k):
        if k not in fmt:
            raise KeyError(f"unknown manifest key {k}")
        return fmt[k].replace("{,}", ",")
    s = re.sub(r"\\nmci\{([^{}]*)\}", lambda m: f"{nm(m.group(1) + '/lo')}–{nm(m.group(1) + '/hi')}", s)
    s = re.sub(r"\\nmdiff\{([^{}]*)\}\{([^{}]*)\}", lambda m: f"{round(raw[m.group(1)] - raw[m.group(2)], 1):.1f}", s)
    s = re.sub(r"\\nmraw\{([^{}]*)\}", lambda m: repr(raw[m.group(1)]), s)
    s = re.sub(r"\\nm\{([^{}]*)\}", lambda m: nm(m.group(1)), s)
    return s


def inline_tex(s: str, labels: dict) -> str:
    """Light LaTeX -> Markdown for table cells and captions (pandoc handles the prose)."""
    s = re.sub(r"\$p_\\text\{ooo\}\s*=\s*([0-9.]+)\$", r"p_ooo = \1", s)      # before \text{} handling (nested braces)
    s = s.replace("\\memzero{}", "Mem0").replace("\\memzero", "Mem0")
    s = re.sub(r"\\begin\{tabular\}\[b\]\{@\{\}c@\{\}\}(.*?)\\\\(.*?)\\end\{tabular\}", r"\1 \2", s)
    s = re.sub(r"\{\\scriptsize\s*(\[[^\]]*\])\}", r"\1", s)
    s = re.sub(r"\\(c|C)?ref\{([^}]*)\}", lambda m: labels.get(m.group(2), "??"), s)
    s = re.sub(r"\\text(it|bf|tt)?\{([^{}]*)\}", lambda m: {"it": f"*{m.group(2)}*", "bf": f"**{m.group(2)}**",
                                                            "tt": f"`{m.group(2)}`"}.get(m.group(1), m.group(2)), s)
    s = re.sub(r"\\emph\{([^{}]*)\}", r"*\1*", s)
    s = re.sub(r"\\multicolumn\{\d+\}\{[^}]*\}\{(.*)\}", r"\1", s)
    s = (s.replace("\\,", " ").replace("~", " ").replace("\\%", "%").replace("\\_", "_").replace("\\&", "&")
         .replace("$\\to$", "→").replace("$\\geq$", "≥").replace("$\\ast$", "*").replace("$^\\ast$", "*")
         .replace("--", "–").replace("\\ ", " ").replace("``", "\"").replace("''", "\""))
    s = re.sub(r"\$p_\\text\{ooo\}\s*=\s*([0-9.]+)\$", r"p_ooo = \1", s).replace("$p_\\text{ooo}", "$p_{ooo}")
    s = re.sub(r"\$([^$]*)\$", lambda m: m.group(1).replace("\\tau", "τ").replace("\\theta", "θ").replace("{-}", "-")
               .replace("\\ldots", "...").replace("\\in\\{", "∈ {").replace("\\}", "}").replace("_\\text{ooo}", "_ooo")
               .replace("\\", ""), s)
    return re.sub(r"\s+", " ", s).strip()


def table_to_md(tex: str, fmt, raw, labels) -> str:
    tex = expand_numbers(tex, fmt, raw)
    cap = re.search(r"\\caption\{(.*)\}\s*\n\\label\{([^}]*)\}", tex, re.S)
    out = [f"**Table {labels.get(cap.group(2), '?')}.** {inline_tex(cap.group(1), labels)}", ""]
    # outer tables only: their \end{tabular} is followed by a newline or the \hspace between side-by-side
    # tables; nested two-line headers end inside a cell (followed by '&', '\\' or '}')
    for tab in re.findall(r"\\begin\{tabular\}\{[^}]*\}\n(.*?)\\end\{tabular\}(?=\s*\n|\\hspace)", tex, re.S):
        rows, header_done, title = [], False, None
        for line in tab.splitlines():
            line = line.strip()
            if not line or line.startswith(("\\toprule", "\\bottomrule", "\\cmidrule")):
                continue
            if line == "\\midrule":
                continue
            line = re.sub(r"\\cmidrule\(lr\)\{[^}]*\}", "", line)
            cells = [inline_tex(c, labels) for c in re.split(r"(?<!\\)&", line.rstrip("\\").rstrip().rstrip("\\"))]
            if len(cells) == 1 and "\\textbf" in line and not rows:
                title = cells[0]
                continue
            rows.append(cells)
        ncol = max(len(r) for r in rows)
        if title:
            out += [f"*{title.strip('*')}*", ""]
        hdr = rows[0] + [""] * (ncol - len(rows[0]))
        body = rows[1:]
        if body and len(body[0]) == ncol and all(c in ("", "p50", "p95") for c in body[0]):   # latency sub-header
            hdr = [f"{h} {s}".strip() for h, s in zip(_spread(hdr, ncol), body[0])]
            body = body[1:]
        out.append("| " + " | ".join(hdr) + " |")
        out.append("|" + "---|" * ncol)
        for r in body:
            out.append("| " + " | ".join(r + [""] * (ncol - len(r))) + " |")
        out.append("")
    return "\n".join(out)


def _spread(hdr, ncol):
    """Expand '\\multicolumn{2}' header cells (already flattened to text) across their columns."""
    out = []
    for h in hdr:
        out += [h, h] if h and h not in ("System",) and len(out) < ncol - 1 else [h]
    return (out + [""] * ncol)[:ncol]


def main() -> int:
    import pypandoc
    fmt, raw = load_numbers()
    labels = load_labels()
    main_tex = (P / "main.tex").read_text(encoding="utf-8")
    title = re.search(r"\\title\{([^}]*)\}", main_tex).group(1)
    tables, figures = {}, {}

    def load_section(name):
        s = (P / f"{name}.tex").read_text(encoding="utf-8")
        s = re.sub(r"(?<!\\)%.*$", "", s, flags=re.M)

        def tab(m):
            if m.group(1).endswith("app_prompts"):        # not a table: verbatim prompt blocks, let pandoc convert
                txt = (P / f"{m.group(1)}.tex").read_text(encoding="utf-8")
                txt = re.sub(r"\\begingroup\\fontsize\{[^}]*\}\{[^}]*\}\\selectfont|\\endgroup", "", txt)
                return re.sub(r"\\(begin|end)\{footnotesize\}", "", txt)
            key = f"TABLEPLACEHOLDER{len(tables)}X"
            tables[key] = table_to_md((P / f"{m.group(1)}.tex").read_text(encoding="utf-8"), fmt, raw, labels)
            return f"\n\n{key}\n\n"
        s = re.sub(r"\\input\{(tables/[^}]*)\}", tab, s)

        def fig(m):
            body = m.group(1)
            img = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{figures/([^}.]*)\.pdf\}", body).group(1)
            cap = re.search(r"\\caption\{(.*)\}\s*\\label\{([^}]*)\}", body, re.S)
            key = f"FIGUREPLACEHOLDER{len(figures)}X"
            figures[key] = (f"![Figure {labels.get(cap.group(2), '?')}](figures/preview/{img}.png)\n\n"
                            f"**Figure {labels.get(cap.group(2), '?')}.** {inline_tex(expand_numbers(cap.group(1), fmt, raw), labels)}")
            return f"\n\n{key}\n\n"
        s = re.sub(r"\\begin\{figure\*?\}(?:\[[^\]]*\])?(.*?)\\end\{figure\*?\}", fig, s, flags=re.S)
        s = expand_numbers(s, fmt, raw)
        s = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", r"\\[\1\\]", s, flags=re.S)   # -> $$...$$
        s = s.replace("\\memzero{}", "Mem0").replace("\\memzero", "Mem0")
        s = re.sub(r"\\(c|C)?ref\{([^}]*)\}", lambda m: labels.get(m.group(2), "??"), s)
        # number the headings like the PDF
        s = re.sub(r"\\section\{([^}]*)\}\s*\\label\{([^}]*)\}",
                   lambda m: f"\\section*{{{labels.get(m.group(2), '')} {m.group(1)}}}", s)
        s = re.sub(r"\\subsection\{([^}]*)\}\s*\\label\{([^}]*)\}",
                   lambda m: f"\\subsection*{{{labels.get(m.group(2), '')} {m.group(1)}}}", s)
        return s

    sections = re.findall(r"\\input\{(sections/[^}]*)\}", main_tex)
    abstract = load_section(sections[0])
    body = "\n\n".join(load_section(s) for s in sections[1:])
    doc = ("\\documentclass{article}\n\\begin{document}\n\\section*{Abstract}\n" + abstract + "\n\n" + body +
           "\n\n\\section*{References}\n\\end{document}\n")
    md = pypandoc.convert_text(doc, "commonmark_x-attributes-fenced_divs-bracketed_spans-smart", format="latex",
                               extra_args=["--citeproc", f"--bibliography={P / 'references.bib'}", "--wrap=none",
                                           "--metadata=link-citations:false", "--metadata=reference-section-title:"])
    for k, v in {**tables, **figures}.items():
        md = md.replace(k, v)
    md = re.sub(r"^</?div[^>]*>\s*$\n?", "", md, flags=re.M)        # citeproc wrappers around the reference list
    leftover = re.findall(r"\\[a-zA-Z]+", re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", "", md, flags=re.S))   # outside math
    head = (f"# {title}\n\n*Anonymous authors. Markdown twin of `paper/main.pdf`, generated by "
            f"`scripts/paper_to_md.py` from the same sources; every number comes from `paper/numbers_manifest.json`.*\n\n")
    (P / "main.md").write_text(head + md, encoding="utf-8", newline="\n")
    print(f"wrote paper/main.md ({len(md.split())} words, {len(tables)} tables, {len(figures)} figures); "
          f"unconverted LaTeX commands: {sorted(set(leftover)) or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
