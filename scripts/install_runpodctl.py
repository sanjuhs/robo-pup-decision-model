#!/usr/bin/env python3
"""Install one pinned Linux amd64 CLI, checking its official release digest."""
import hashlib
from pathlib import Path
import platform
import urllib.request

VERSION = "v3.0.1"
SHA256 = "b66d11eca8cacbf18a2de6943c6e18a79ffbdc8a3c30aa13a49dc6cfc15d4239"
DEST = Path("/workspace/.tools/runpodctl")


def main():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise SystemExit("This installer is pinned for Linux amd64 only")
    if DEST.exists() and hashlib.sha256(DEST.read_bytes()).hexdigest() == SHA256:
        print("runpodctl v3.0.1 already verified")
        return
    base = f"https://github.com/runpod/runpodctl/releases/download/{VERSION}/"
    with urllib.request.urlopen(base+"checksums_3.0.1_sha256.txt", timeout=60) as response:
        checksums = response.read().decode()
    expected = next(line.split()[0] for line in checksums.splitlines()
                    if line.split()[1:] == ["runpodctl-linux-amd64"])
    if expected != SHA256:
        raise SystemExit("Official checksum differs from audited pin; manual review required")
    with urllib.request.urlopen(base+"runpodctl-linux-amd64", timeout=60) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise SystemExit("Checksum verification failed; binary not installed")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    temp = DEST.with_suffix(".download")
    temp.write_bytes(data)
    temp.chmod(0o755)
    temp.replace(DEST)
    print("runpodctl v3.0.1 installed; SHA-256 verified")


if __name__ == "__main__":
    main()
