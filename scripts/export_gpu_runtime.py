#!/usr/bin/env python3
"""Keep locked hashes for all dependencies, retaining the provider CUDA PyTorch build."""

import argparse
from pathlib import Path
import subprocess


def without_cpu_torch(requirements):
    lines = []
    skipping = False
    for line in requirements.splitlines():
        if line.startswith("torch=="):
            skipping = True
            continue
        if skipping and line.startswith((" ", "\t")):
            continue
        skipping = False
        lines.append(line)
    output = "\n".join(lines) + "\n"
    if "torch==" in output or "--hash=sha256:" not in output:
        raise ValueError("Unexpected dependency export")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = subprocess.run(
        ["uv", "export", "--frozen", "--no-emit-project", "--no-annotate"],
        capture_output=True,
        text=True,
        check=True,
    )
    with a.out.open("x") as stream:
        stream.write(without_cpu_torch(result.stdout))
    print(
        "Exported hash-locked non-Torch runtime; provider GPU Torch is an explicit recorded exception"
    )
