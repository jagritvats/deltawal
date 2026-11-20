"""Double-blind / secret audit of everything that ships: git-tracked files, the paper/ build outputs
(gzip snapshot decompressed, PDF and PNG bytes including metadata), and commit metadata.

Identity terms are read at runtime (OS user name, the global git identity, home directory) and secret
values from .env, so this script itself contains nothing personal. Prints only the file and the rule
that matched, never the matched secret. Exit code 1 on any finding.

  uv run python scripts/audit_anonymity.py
"""
from __future__ import annotations

import getpass
import gzip
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GENERIC = {
    "email address": re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z0-9.-]*[A-Za-z]{2,}"),
    "windows user path": re.compile(rb"[A-Za-z]:[\\/]+Users[\\/]+", re.I),
    "drive-letter absolute path": re.compile(rb"\b[A-Za-z]:[\\/]{1,2}(dev|tmp|work|src|projects)\b", re.I),
    "unix home path": re.compile(rb"(/home/|/Users/)[A-Za-z0-9_.-]+"),
    "api-key-like token": re.compile(rb"\b(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"),
    "AI attribution": re.compile(rb"Co-Authored-By|Generated with Claude|Claude-Session", re.I),
}
# public, non-identifying addresses that legitimately appear in vendored metadata / docs
ALLOWED_EMAILS = {b"noreply@github.com"}


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout


def identity_terms() -> dict[str, bytes]:
    terms = {}
    for label, val in (("os user name", getpass.getuser()), ("global git name", _git("config", "--global", "user.name").strip()),
                       ("global git email", _git("config", "--global", "user.email").strip()),
                       ("home directory", str(Path.home()))):
        if val and len(val) >= 3:
            terms[label] = val.encode()
            if label == "global git email":
                terms["global git email local part"] = val.split("@")[0].encode()
    secrets = {}
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if len(v) >= 12:
                    secrets[f".env value of {k.strip()}"] = v.encode()
    return {**terms, **secrets}


def payloads(path: Path):
    data = path.read_bytes()
    yield data
    if path.suffix == ".gz":
        yield gzip.decompress(data)
    if path.suffix == ".pdf":           # FlateDecode streams: decompress and scan text too
        import zlib
        for m in re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S):
            try:
                yield zlib.decompress(m.group(1))
            except zlib.error:
                pass


def main() -> int:
    tracked = [ROOT / p for p in _git("ls-files", "-z").split("\0") if p]
    built = [p for p in (ROOT / "paper").rglob("*") if p.is_file()]
    staged_new = [ROOT / p for p in _git("ls-files", "-z", "--others", "--exclude-standard").split("\0") if p]
    files = sorted({p.resolve() for p in tracked + built + staged_new if p.exists()})
    terms = identity_terms()
    findings = []
    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        for blob in payloads(f):
            for label, rx in GENERIC.items():
                for m in rx.finditer(blob):
                    if label == "email address" and m.group(0) in ALLOWED_EMAILS:
                        continue
                    if label == "AI attribution" and rel == "scripts/audit_anonymity.py":   # the rule itself
                        continue
                    findings.append((rel, label))
                    break
            low = blob.lower()
            for label, t in terms.items():
                if t.lower() in low:
                    findings.append((rel, label))
    meta = _git("log", "--all", "--format=%an <%ae> | %cn <%ce>%n%B")
    for label, t in terms.items():
        if t.lower() in meta.lower().encode():
            findings.append(("git commit metadata", label))
    if GENERIC["AI attribution"].search(meta.encode()):
        findings.append(("git commit metadata", "AI attribution"))
    authors = sorted(set(_git("log", "--all", "--format=%an <%ae>").splitlines()))
    findings = sorted(set(findings))
    print(f"scanned {len(files)} files (tracked + untracked-not-ignored + paper/ outputs) and {len(meta.splitlines())} "
          f"lines of commit metadata; identity terms checked: {sorted(terms)}")
    print(f"commit authors: {authors}")
    for rel, label in findings:
        print(f"FINDING  {rel}: {label}")
    print("clean" if not findings else f"{len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
