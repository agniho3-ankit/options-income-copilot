#!/usr/bin/env python3
"""Fail if any file that would be committed contains PII, secrets, or private material.

Stdlib only, so it runs as a pre-commit hook before any dependency is installed.

Usage:
    python scripts/privacy_audit.py            # every file git would commit (tracked + untracked, not ignored)
    python scripts/privacy_audit.py --staged   # only files staged for the next commit
    python scripts/privacy_audit.py FILE...    # explicit files (pre-commit passes these)

Checks:
    - email addresses, except GitHub noreply and example domains
    - absolute home paths (macOS, Linux, Windows), which leak the OS username
    - strings that look like API keys or long random tokens
    - entries in the gitignored .privacy_denylist file (name, username, account numbers)
    - non-empty secret values from the local .env (SEC_USER_AGENT, *_KEY, *_TOKEN, *_SECRET)
    - forbidden files: .env, the denylist, documents, databases, model weights, data folders

Findings print as `path:line: rule` and never echo the matched text.
A line containing `privacy-audit: allow` is skipped (use sparingly, after review).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DENYLIST_FILE = ".privacy_denylist"
ALLOW_MARKER = "privacy-audit: allow"

ALLOWED_EMAIL_DOMAINS = ("users.noreply.github.com", "example.com", "example.org", "example.net")
ALLOWED_EMAILS = {"noreply@anthropic.com", "git@github.com"}

FORBIDDEN_NAMES = {".env", DENYLIST_FILE}
FORBIDDEN_SUFFIXES = {
    ".docx", ".doc", ".pdf", ".xlsx", ".db", ".sqlite", ".sqlite3",
    ".parquet", ".pkl", ".onnx", ".bin", ".safetensors", ".pem", ".key",
}
FORBIDDEN_DIRS = {"data", "demo_outputs", ".venv", ".claude"}

# Used only when the folder is not a git repo yet; mirrors .gitignore.
WALK_SKIP_DIRS = {".git", ".venv", "data", "demo_outputs", "__pycache__", ".pytest_cache",
                  ".ruff_cache", ".mypy_cache", "build", "dist"}
WALK_SKIP_FILES = {".env", DENYLIST_FILE, ".DS_Store"}

ENV_SECRET_KEY = re.compile(r"^(SEC_USER_AGENT|.*_(KEY|TOKEN|SECRET|PASSWORD))$")

# Character classes such as "[r]" keep these patterns from matching their own source text.
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PATTERNS = [
    ("home-path", re.compile(r"/Use[r]s/[A-Za-z0-9._-]+|/hom[e]/[A-Za-z0-9._-]+|[A-Za-z]:\\+Use[r]s\\+")),
    ("anthropic-key", re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}")),
    ("openai-key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}")),
    ("github-token", re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}")),
    ("aws-key", re.compile(r"AKI[A][0-9A-Z]{16}")),
    ("long-hex-token", re.compile(r"\b[0-9a-fA-F]{40,}\b")),
]
LONG_TOKEN = re.compile(r"[A-Za-z0-9+/_-]{40,}={0,2}")


@dataclass(frozen=True)
class Finding:
    path: str
    line: int  # 0 means the finding is about the file itself
    rule: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}" if self.line else f"{self.path}: {self.rule}"


def _email_allowed(address: str) -> bool:
    address = address.lower()
    if address in ALLOWED_EMAILS:
        return True
    domain = address.rsplit("@", 1)[-1]
    return any(domain == d or domain.endswith("." + d) for d in ALLOWED_EMAIL_DOMAINS)


def _looks_random(token: str) -> bool:
    """Long tokens count only if they mix upper, lower and digits (skips identifiers and words)."""
    return (any(c.isupper() for c in token) and any(c.islower() for c in token)
            and any(c.isdigit() for c in token))


def scan_text(text: str, path: str, denylist: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    lowered_deny = [d.lower() for d in denylist]
    for lineno, line in enumerate(text.splitlines(), start=1):
        if ALLOW_MARKER in line:
            continue
        if any(not _email_allowed(m) for m in EMAIL.findall(line)):
            findings.append(Finding(path, lineno, "email"))
        for rule, pattern in PATTERNS:
            if pattern.search(line):
                findings.append(Finding(path, lineno, rule))
        if any(_looks_random(t) for t in LONG_TOKEN.findall(line)):
            findings.append(Finding(path, lineno, "long-random-token"))
        low = line.lower()
        for idx, entry in enumerate(lowered_deny, start=1):
            if entry in low:
                findings.append(Finding(path, lineno, f"denylist entry #{idx}"))
    return findings


def check_path(rel: str, denylist: list[str]) -> list[Finding]:
    """Rules about the file name and location, independent of content."""
    findings: list[Finding] = []
    p = Path(rel)
    if p.name in FORBIDDEN_NAMES or (p.name.startswith(".env") and p.name != ".env.example"):
        findings.append(Finding(rel, 0, "forbidden file (secrets/denylist)"))
    if p.suffix.lower() in FORBIDDEN_SUFFIXES:
        findings.append(Finding(rel, 0, f"forbidden file type {p.suffix.lower()}"))
    if p.parts and p.parts[0] in FORBIDDEN_DIRS:
        findings.append(Finding(rel, 0, f"file under {p.parts[0]}/ (local data only)"))
    if p.name.endswith((".local.yaml", ".local.yml")):
        findings.append(Finding(rel, 0, "owner-local file (labels or settings)"))
    if "eval_snapshots" in p.parts:
        findings.append(Finding(rel, 0, "evaluation snapshot (local data only)"))
    low = rel.lower()
    for idx, entry in enumerate(denylist, start=1):
        if entry.lower() in low:
            findings.append(Finding(rel, 0, f"denylist entry #{idx} in file name"))
    return findings


def load_denylist(root: Path) -> list[str]:
    entries: list[str] = []
    deny = root / DENYLIST_FILE
    if deny.is_file():
        for raw in deny.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    env = root / ".env"
    if env.is_file():
        for raw in env.read_text(encoding="utf-8").splitlines():
            key, sep, value = raw.partition("=")
            value = value.strip().strip('"').strip("'")
            if sep and ENV_SECRET_KEY.match(key.strip()) and len(value) >= 4:
                entries.append(value)
    return entries


def _git(root: Path, *args: str) -> list[str] | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]


def list_candidate_files(root: Path, staged: bool) -> list[str]:
    if staged:
        files = _git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    else:
        files = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if files is not None:
        return sorted(files)
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in WALK_SKIP_DIRS and not d.endswith(".egg-info")]
        for name in filenames:
            if name not in WALK_SKIP_FILES:
                found.append(str((Path(dirpath) / name).relative_to(root)))
    return sorted(found)


def audit(root: Path, files: list[str], denylist: list[str]) -> list[Finding]:
    findings: list[Finding] = []
    for rel in files:
        findings.extend(check_path(rel, denylist))
        full = root / rel
        if not full.is_file():
            continue
        data = full.read_bytes()
        if b"\0" in data[:8192]:
            continue  # binary; its type was already checked by name
        findings.extend(scan_text(data.decode("utf-8", "replace"), rel, denylist))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="files to scan (default: everything git would commit)")
    parser.add_argument("--staged", action="store_true", help="scan only staged files")
    args = parser.parse_args(argv)

    denylist = load_denylist(REPO_ROOT)
    files = args.files or list_candidate_files(REPO_ROOT, args.staged)
    files = [os.path.relpath(Path(f).resolve(), REPO_ROOT) if Path(f).is_absolute() else f for f in files]
    findings = audit(REPO_ROOT, files, denylist)

    if not (REPO_ROOT / DENYLIST_FILE).is_file():
        print(f"warning: {DENYLIST_FILE} not found; personal-name checks are off", file=sys.stderr)
    for f in findings:
        print(f)
    print(f"privacy audit: {len(files)} files scanned, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
