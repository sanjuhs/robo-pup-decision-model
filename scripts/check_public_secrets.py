#!/usr/bin/env python3
"""Check tracked content (and optionally history) without printing secret values."""
import argparse
from pathlib import Path
import re
import subprocess

PATTERN = re.compile(
    rb"rpa_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    rb"github_pat_[A-Za-z0-9_]{30,}|"
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)


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
            print("Potential credential in tracked file: " + str(file.relative_to(root)))
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
