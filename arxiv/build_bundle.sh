#!/usr/bin/env bash
# Build the arXiv source bundle for the preprint (arxiv/arxiv_bundle.tar.gz) and verify it compiles from scratch.
#
#   bash arxiv/build_bundle.sh
#
# Steps:
#   1. stage only what arXiv needs into arxiv/build/src: main.tex, numbers.tex, sections/, tables/, the four
#      figure PDFs, and the style files flattened to the top level (arXiv does not honour TEXINPUTS);
#   2. compile a scratch copy (with references.bib + tmlr.bst) in the pinned TeX Live image to produce main.bbl,
#      and add main.bbl to the stage (arXiv does not run bibtex reliably; it uses the shipped .bbl);
#   3. pack a deterministic tarball (sorted names, fixed mtime, numeric owner 0);
#   4. extract it into an empty directory and run pdflatex three times there without bibtex (labels settle after the bibliography); fail on any LaTeX
#      error, undefined reference/citation, local path, or TMLR review/branding string.
# No numbers are regenerated: the paper's numbers.tex, tables and figures are copied as they are.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
OUT="$ROOT/arxiv"
B="$OUT/build"
TEXLIVE_IMAGE="texlive/texlive@sha256:c572bf2d14631545769e5a8f8d315e7b8f8896b14720b1c7a063917d503fd46a"
docker_dir() { (cd "$1" && (pwd -W 2>/dev/null || pwd)); }

rm -rf "$B"
mkdir -p "$B/src/sections" "$B/src/tables" "$B/src/figures" "$B/work" "$B/verify"

# 1. stage
cp paper/main.tex paper/numbers.tex "$B/src/"
cp paper/sections/*.tex "$B/src/sections/"
cp paper/tables/*.tex "$B/src/tables/"
for f in $(grep -ho 'includegraphics\(\[[^]]*\]\)\?{[^}]*}' paper/sections/*.tex | sed 's/.*{\(.*\)}/\1/' | sort -u); do
  cp "paper/$f" "$B/src/$f"
done
cp paper/style/tmlr.sty paper/style/fancyhdr.sty paper/style/LICENSE-tmlr-style "$B/src/"

# 2. compile a scratch copy to get main.bbl
cp -r "$B/src/." "$B/work/"
cp paper/references.bib paper/style/tmlr.bst "$B/work/"
MSYS_NO_PATHCONV=1 docker run --rm -v "$(docker_dir "$B/work"):/w" -w /w "$TEXLIVE_IMAGE" sh -c '
  run() { pdflatex -interaction=nonstopmode -halt-on-error main.tex > stdout.txt || { grep -A6 "^!" main.log | head -40; exit 1; }; }
  run; bibtex main > bibtex.txt 2>&1 || { cat bibtex.txt; exit 1; }; run; run'
cp "$B/work/main.bbl" "$B/src/"

# 3. pack (deterministic)
find "$B/src" -name '.*' -prune -exec rm -rf {} +
tar --sort=name --mtime='2026-11-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
    -C "$B/src" -czf "$OUT/arxiv_bundle.tar.gz" .

# 4. verify from a clean extraction: pdflatex twice, no bibtex
tar -xzf "$OUT/arxiv_bundle.tar.gz" -C "$B/verify"
MSYS_NO_PATHCONV=1 docker run --rm -v "$(docker_dir "$B/verify"):/w" -w /w "$TEXLIVE_IMAGE" sh -c '
  mkdir -p /tmp/v && cp -r /w/. /tmp/v/ && cd /tmp/v
  for i in 1 2 3; do pdflatex -interaction=nonstopmode -halt-on-error main.tex > stdout.txt \
      || { grep -A6 "^!" main.log | head -40; exit 1; }; done
  fail=0
  grep -nE "Reference .* undefined|Citation .* undefined|There were undefined|Rerun to get" main.log && fail=1
  grep -nE "Overfull \\hbox \(([1-9]|[0-9]{2,})" main.log && fail=1
  echo "pages: $(grep -o "[0-9]* pages" main.log | tr -dc 0-9)"
  cp main.pdf main.log /w/
  exit $fail'

fail=0
if grep -rIlE '[A-Za-z]:[\\/](Users|dev|tmp)|/Users/|/home/' "$B/src"; then echo "local path found"; fail=1; fi
if grep -rIlE 'Transactions on Machine Learning|Under review|double-blind' "$B/src" --exclude=tmlr.sty; then
  echo "review/branding string found"; fail=1; fi
# openreview.net may appear only as a cited paper's URL in the bibliography, never in the paper's own text
if grep -rIlE 'openreview\.net' "$B/src" --exclude=tmlr.sty --exclude=main.bbl; then
  echo "openreview link outside the bibliography"; fail=1; fi
echo "bundle: $OUT/arxiv_bundle.tar.gz ($(du -h "$OUT/arxiv_bundle.tar.gz" | cut -f1), $(tar -tzf "$OUT/arxiv_bundle.tar.gz" | grep -vc '/$') files)"
exit $fail
