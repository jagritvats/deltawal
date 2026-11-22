#!/usr/bin/env bash
# Compile paper/main.tex (preprint, TMLR style) with a pinned TeX Live image. Fails on LaTeX errors
# (including unknown manifest keys), undefined references/citations, or overfull boxes >= 1pt.
# Output: paper/main.pdf, paper/main.log.
set -euo pipefail
cd "$(dirname "$0")"
TEXLIVE_IMAGE="texlive/texlive@sha256:c572bf2d14631545769e5a8f8d315e7b8f8896b14720b1c7a063917d503fd46a"
HOSTDIR="$(pwd -W 2>/dev/null || pwd)"
MSYS_NO_PATHCONV=1 docker run --rm -v "$HOSTDIR:/w" -w /w -e TEXINPUTS=/w/style: -e BSTINPUTS=/w/style: \
    -e BIBINPUTS=/w: "$TEXLIVE_IMAGE" sh -c '
  mkdir -p /tmp/b
  run() { pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp/b main.tex > /tmp/b/stdout.txt \
          || { grep -A6 "^!" /tmp/b/main.log | head -40; exit 1; }; }
  run
  (cd /tmp/b && bibtex main > /tmp/b/bibtex.txt 2>&1) || grep -E "error|Warning" /tmp/b/bibtex.txt || true
  run; run
  cp /tmp/b/main.pdf /tmp/b/main.log /tmp/b/main.aux /w/
  fail=0
  grep -E "^Warning--" /tmp/b/bibtex.txt && fail=1
  grep -nE "Overfull \\hbox \(([1-9]|[0-9]{2,})" /tmp/b/main.log && fail=1
  grep -nE "Reference .* undefined|Citation .* undefined|There were undefined" /tmp/b/main.log && fail=1
  echo "pages: $(grep -o "[0-9]* pages" /tmp/b/main.log | tr -dc 0-9)"
  exit $fail'

# Markdown twin for internal reading / tool context (same numbers, same Table/Figure numbering as the PDF)
cd .. && uv run --quiet --with pypandoc_binary python scripts/paper_to_md.py
