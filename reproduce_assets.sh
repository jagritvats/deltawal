#!/usr/bin/env bash
# Rebuild every paper table, figure and the numbers manifest (deterministic, byte-identical across runs).
#
#   bash reproduce_assets.sh               # from the committed snapshot paper/data/snapshot.json.gz (no data,
#                                          # models or infrastructure needed)
#   bash reproduce_assets.sh --collect     # first re-collect the snapshot from runs/ (needs the run artifacts)
#   bash reproduce_assets.sh --check-tex   # also compile all tables + figures at TMLR text width with TeX Live
#                                          # (docker) and fail on any overfull/underfull box or LaTeX warning
set -euo pipefail
cd "$(dirname "$0")"

TEXLIVE_IMAGE="texlive/texlive@sha256:c572bf2d14631545769e5a8f8d315e7b8f8896b14720b1c7a063917d503fd46a"
COLLECT=0
TEX=0
for a in "$@"; do
  case "$a" in
    --collect) COLLECT=1 ;;
    --check-tex) TEX=1 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

uv sync --group paper --quiet
if [ "$COLLECT" = 1 ]; then
  uv run python scripts/build_paper_assets.py
else
  uv run python scripts/build_paper_assets.py --from-snapshot
fi

if [ "$TEX" = 1 ]; then
  HOSTDIR="$(pwd -W 2>/dev/null || pwd)/paper"          # pwd -W: Windows path under Git Bash
  MSYS_NO_PATHCONV=1 docker run --rm -v "$HOSTDIR:/w" -w /w "$TEXLIVE_IMAGE" sh -c '
    set -e
    for pass in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error -output-directory=/tmp check_assets.tex >/dev/null; done
    cp /tmp/check_assets.pdf /w/check_assets.pdf
    if grep -nE "Overfull|Underfull|Warning" /tmp/check_assets.log; then echo "LaTeX layout warnings" >&2; exit 1; fi
    echo "check_assets.tex: $(grep -o "Output written on [^ ]* ([0-9]* pages" /tmp/check_assets.log), no warnings"'
fi
