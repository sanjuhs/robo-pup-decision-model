#!/usr/bin/env python3
"""Explicitly download a revision-pinned public VLM; no token is normally needed.

This creates a local checkpoint, not a trained model. The Hugging Face token, if
the operator has configured one for rate limits, stays in process memory.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import quote
from urllib.request import Request, urlopen

from robopup.vlm_runtime import (
    DEFAULT_MODEL, DEFAULT_MODEL_DIR, DEFAULT_MODEL_REVISION, MANIFEST_NAME,
)


def resolve_revision(model_id, revision):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        raise ValueError("Use an owner/model Hugging Face model ID")
    url = f"https://huggingface.co/api/models/{model_id}/revision/{quote(revision, safe='')}"
    with urlopen(Request(url, headers={"User-Agent": "RoboPup-Arena-explicit-download"}),
                 timeout=30) as response:
        record = json.load(response)
    sha = record.get("sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Hugging Face did not return an immutable commit revision")
    return sha, (record.get("cardData") or {}).get("license", "not-reported")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="Default Qwen revision is pinned; other models use main")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        parser.exit(1, "Install the optional VLM dependencies first: uv sync --extra vlm\n")
    # No token arguments or environment values are printed or persisted.
    selected_revision = args.revision or (
        DEFAULT_MODEL_REVISION if args.model == DEFAULT_MODEL else "main"
    )
    revision, license_id = resolve_revision(args.model, selected_revision)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / MANIFEST_NAME
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing.get("revision") != revision or existing.get("model_id") != args.model:
            parser.exit(1, "Output contains another checkpoint revision; choose a new directory\n")
    snapshot_download(
        repo_id=args.model, revision=revision, local_dir=args.output,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja",
                        "LICENSE*", "README.md", "tokenizer*", "merges.txt", "vocab.json"],
    )
    files = [path for path in sorted(args.output.rglob("*"))
             if path.is_file() and ".cache" not in path.relative_to(args.output).parts
             and path.name != MANIFEST_NAME]
    if not (args.output / "config.json").is_file() or not any(
        path.suffix == ".safetensors" for path in files
    ):
        parser.exit(1, "Download did not produce a config and safetensors checkpoint\n")
    record = {
        "schema": "robopup-vlm-checkpoint-v1", "model_id": args.model,
        "revision": revision, "license": license_id,
        "source": f"https://huggingface.co/{args.model}/tree/{revision}",
        "trained_by_robopup": False,
        "files": [{"path": str(path.relative_to(args.output)),
                   "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }
    temporary = args.output / (MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(args.output / MANIFEST_NAME)
    print(json.dumps({"model": args.model, "revision": revision, "license": license_id,
                      "files": len(files), "bytes": sum(file["bytes"] for file in record["files"]),
                      "manifest": str(args.output / MANIFEST_NAME)}))


if __name__ == "__main__":
    main()
