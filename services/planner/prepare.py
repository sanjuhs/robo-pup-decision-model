"""Bundle the exact two lightweight core modules; never bundle robot dependencies."""

import hashlib
from pathlib import Path
import shutil


def prepare():
    service = Path(__file__).resolve().parent
    source = service.parents[1] / "src" / "robopup"
    destination = service / "robopup"
    destination.mkdir(exist_ok=True)
    (destination / "__init__.py").write_text('"""Lightweight mission protocol/runtime bundle."""\n')
    for name in ("vlm_protocol.py", "vlm_runtime.py"):
        shutil.copyfile(source / name, destination / name)
        digest = hashlib.sha256((destination / name).read_bytes()).hexdigest()
        print(f"Bundled {name}: {digest}")


if __name__ == "__main__":
    prepare()
