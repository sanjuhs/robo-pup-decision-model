#!/usr/bin/env python3
"""Export locked wheel URLs for the tested Linux x86_64 / Python 3.12 pod ABI.

No index lookup or dependency resolution is needed inside the pod. URLs/hashes
come from uv.lock; CUDA Torch remains the separately recorded provider build.
"""

import argparse
from pathlib import Path
import sys
import tomllib
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

from export_gpu_runtime import without_cpu_torch
import subprocess


def export(out):
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux":
        raise RuntimeError("Generate on tested Linux Python 3.12 host")
    root = Path(__file__).resolve().parents[1]
    locked = tomllib.loads((root / "uv.lock").read_text())
    result = subprocess.run(
        ["uv", "export", "--frozen", "--no-emit-project", "--no-annotate"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    requirements = without_cpu_torch(result.stdout)
    ranks = {tag: i for i, tag in enumerate(sys_tags())}
    entries = []
    for line in requirements.splitlines():
        if not line or line.startswith(("#", " ", "\t")):
            continue
        req = Requirement(line.rstrip("\\").strip())
        if req.marker and not req.marker.evaluate(default_environment()):
            continue
        packages = [
            p
            for p in locked["package"]
            if canonicalize_name(p["name"]) == canonicalize_name(req.name)
            and p["version"] in req.specifier
        ]
        candidates = []
        for package in packages:
            for wheel in package.get("wheels", []):
                filename = unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
                _, _, _, tags = parse_wheel_filename(filename)
                best = min((ranks[t] for t in tags if t in ranks), default=10**9)
                if best < 10**9:
                    candidates.append((best, wheel))
        if not candidates:
            raise ValueError("No locked wheel for " + req.name)
        wheel = min(candidates, key=lambda x: x[0])[1]
        if urlsplit(wheel["url"]).hostname != "files.pythonhosted.org":
            raise ValueError("Unexpected wheel host")
        entries.append(f"{req.name} @ {wheel['url']} --hash={wheel['hash']}")
    with out.open("x") as stream:
        stream.write(
            "# Hash-locked direct wheels: Linux x86_64, CPython 3.12; Torch from provider image.\n"
            + "\n".join(entries)
            + "\n"
        )
    print(f"Exported {len(entries)} direct wheel URLs with lockfile hashes")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    export(a.out)
