#!/usr/bin/env python3
"""Check selected credential patterns in tracked content and optional Git history.

This is a small publication guard, not a complete secret-detection service. It
does not recognize arbitrary credentials, legacy unprefixed tokens, or JWTs.
Matched values are never printed or checked against a provider's API.
"""
import argparse
from pathlib import Path
import re
import subprocess

PATTERN = re.compile(
    rb"rpa_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    rb"github_pat_[A-Za-z0-9_]{30,}|"
    rb"\bhf_[A-Za-z0-9]{34}\b|"
    rb"\bvcp_[A-Za-z0-9_-]{24,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
# HF's current user-token shape is also recognized by established public
# detectors. Vercel documents its current PAT prefix at:
# https://vercel.com/docs/accounts/create-an-access-token
# Only long prefixed values are selected; short documentation placeholders and
# generic unprefixed identifiers intentionally remain outside this check.


def git(*args):
    return subprocess.check_output(["git", *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also scan reachable Git blobs")
    args = parser.parse_args()
    root = Path(git("rev-parse", "--show-toplevel").decode().strip())
    files = [root / p.decode() for p in git("ls-files", "-z").split(b"\0") if p]
    findings = 0
    for file in files:
        if file.is_file() and PATTERN.search(file.read_bytes()):
            path = str(file.relative_to(root)).encode("utf-8", errors="replace")
            safe_path = PATTERN.sub(b"[REDACTED]", path).decode("utf-8", errors="replace")
            print("Potential credential in tracked file: " + safe_path)
            findings += 1
    blobs = 0
    if args.history:
        for line in git("rev-list", "--objects", "--all").splitlines():
            oid = line.split(b" ", 1)[0].decode()
            if git("cat-file", "-t", oid).strip() != b"blob":
                continue
            blobs += 1
            if PATTERN.search(git("cat-file", "blob", oid)):
                print("Potential credential in history blob: " + oid)
                findings += 1
    print(f"Scanned {len(files)} tracked paths and {blobs} history blobs; {findings} findings.")
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
