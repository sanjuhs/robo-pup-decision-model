#!/usr/bin/env python3
"""Verify and seed the two bundled policies without replacing local runs."""
import argparse
import hashlib
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ROOT/"artifacts")
    args = parser.parse_args()
    for source_name, participant, run in (("curious-v2","baseline","curious-v2"),
                                          ("5090-once","runpod","5090-once")):
        source = ROOT/"demo"/source_name
        if not source.exists():
            continue
        expected = (source/"SHA256SUMS").read_text().split()[0]
        if hashlib.sha256((source/"policy.zip").read_bytes()).hexdigest() != expected:
            raise SystemExit("Bundled checkpoint checksum mismatch: "+source_name)
        destination = args.artifact_root/participant/run
        if destination.exists():
            print("Preserved local run: "+participant+"/"+run)
        else:
            shutil.copytree(source,destination)
            print("Seeded verified demo: "+participant+"/"+run)


if __name__ == "__main__":
    main()
