#!/usr/bin/env python3
"""Export locked VLM wheels, retaining the recorded provider CUDA Torch/torchvision."""

import argparse
from pathlib import Path
import subprocess
import sys
import tomllib
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


def export(out):
    if sys.version_info[:2] != (3, 12) or sys.platform != "linux":
        raise RuntimeError("Generate on the tested Linux Python 3.12 host")
    root = Path(__file__).resolve().parents[1]
    locked = tomllib.loads((root / "uv.lock").read_text())
    result = subprocess.run(
        ["uv", "export", "--frozen", "--extra", "vlm", "--no-dev", "--no-emit-project",
         "--no-annotate"], capture_output=True, text=True, check=True, cwd=root,
    )
    ranks = {tag: i for i, tag in enumerate(sys_tags())}
    entries = []
    for line in result.stdout.splitlines():
        if not line or line.startswith(("#", " ", "\t")):
            continue
        req = Requirement(line.rstrip("\\").strip())
        if canonicalize_name(req.name) in ("torch", "torchvision"):
            continue
        if req.marker and not req.marker.evaluate(default_environment()):
            continue
        candidates = []
        for package in locked["package"]:
            if (canonicalize_name(package["name"]) != canonicalize_name(req.name)
                    or package["version"] not in req.specifier):
                continue
            for wheel in package.get("wheels", []):
                filename = unquote(urlsplit(wheel["url"]).path.rsplit("/", 1)[-1])
                _, _, _, tags = parse_wheel_filename(filename)
                rank = min((ranks[tag] for tag in tags if tag in ranks), default=10**9)
                if rank < 10**9:
                    candidates.append((rank, wheel))
        if not candidates:
            raise ValueError("No locked wheel for " + req.name)
        wheel = min(candidates, key=lambda item: item[0])[1]
        if urlsplit(wheel["url"]).hostname != "files.pythonhosted.org":
            raise ValueError("Unexpected wheel host")
        entries.append(f"{req.name} @ {wheel['url']} --hash={wheel['hash']}")
    with out.open("x") as stream:
        stream.write(
            "# Frozen VLM/runtime wheels: Linux x86_64 CPython3.12.\n"
            "# Provider exceptions: torch2.9.0+cu129, torchvision0.24.0+cu129 (checked by job).\n"
            + "\n".join(entries) + "\n"
        )
    print(f"Exported {len(entries)} hash-locked direct wheel URLs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    export(parser.parse_args().out)
