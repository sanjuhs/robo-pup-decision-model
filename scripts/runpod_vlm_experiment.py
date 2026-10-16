#!/usr/bin/env python3
"""Authorized, bounded RTX5090 real VLM inference with memory-only account key.

Creates at most one paid pod, never retries allocation or touches other pods.
The account credential stays in this monitor and the local CLI environment.
"""

import argparse
import base64
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TTL = 25 * 60
MAX_RATE = 1.5
ALLOCATION = 2.0
MAX_INPUT_BYTES = 2 * 1024 * 1024


def pack_inputs(directory):
    """Only PNGs and one JSON manifest are accepted, before any paid allocation."""
    directory = directory.resolve()
    manifest = directory / "prompts.json"
    data = json.loads(manifest.read_text())
    cases = data if isinstance(data, list) else data["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 24:
        raise ValueError("Input manifest must have 1 to 24 cases")
    filenames = {"prompts.json"}
    identifiers = set()
    for index, case in enumerate(cases):
        identifier = case.get("id", f"case-{index:03}")
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identifier)
                or identifier in identifiers):
            raise ValueError("Every case needs a unique bounded ID")
        identifiers.add(identifier)
        filename = case["image"]
        if (not isinstance(filename, str) or Path(filename).name != filename
                or not filename.endswith(".png")):
            raise ValueError("Case image must be a PNG filename without directories")
        from robopup.vlm_protocol import build_planning_prompt
        build_planning_prompt(case["instruction"], energy=case.get("energy", .8),
                              held=case.get("held", False))
        filenames.add(filename)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for filename in sorted(filenames):
            path = directory / filename
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
                raise ValueError("Input files must be bounded, regular files")
            if filename.endswith(".png"):
                from PIL import Image
                with Image.open(path) as image:
                    if (image.format != "PNG" or min(image.size) < 16
                            or image.width * image.height > 16_777_216):
                        raise ValueError("Input image is outside supported PNG bounds")
                    image.verify()
            archive.add(path, arcname="vlm-input/" + filename, recursive=False)
    packed = stream.getvalue()
    if len(packed) > MAX_INPUT_BYTES:
        raise ValueError("Compressed camera inputs exceed 2 MiB")
    return packed, len(cases)


def run(out, input_dir):
    packed, count = pack_inputs(input_dir)
    ref = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    # Use exactly the reviewed committed script, not a changed working copy.
    bootstrap = subprocess.check_output(
        ["git", "-C", str(ROOT), "show", ref + ":scripts/gpu_vlm_job.sh"]
    )
    remote_ref = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-remote", "origin", "refs/heads/main"], text=True,
    ).split()[0]
    if remote_ref != ref:
        raise RuntimeError("Push the reviewed source commit to main before allocating a pod")
    if not (ROOT / "configs/gpu-vlm-runtime-direct-cp312.txt").is_file():
        raise RuntimeError("Frozen VLM wheel manifest is missing")
    if TTL / 3600 * MAX_RATE + .2 > ALLOCATION or ALLOCATION + 1 > 30:
        raise RuntimeError("Experiment budget bounds are inconsistent")
    out.mkdir(parents=True, exist_ok=False)
    key = os.environ.get("RUNPOD_API_KEY") or getpass.getpass(
        "Runpod credential (hidden; memory only): "
    )
    if not key:
        raise ValueError("Runpod credential is missing")
    record = {
        "source_ref": ref, "gpu": "RTX 5090", "experiment": "real-generative-vlm-inference",
        "ttl_seconds": TTL, "total_session_budget_usd": 30,
        "prior_spend_reserve_usd": 1, "experiment_allocation_usd": ALLOCATION,
        "maximum_gpu_rate_usd_hour": MAX_RATE, "storage_reserve_usd": .2,
        "input_cases": count, "input_archive_bytes": len(packed),
        "input_archive_sha256": hashlib.sha256(packed).hexdigest(),
        "paid_scope": "One inference pod; no training, retries, fallback GPU, or network volume",
        "account_key_passed_to_pod": False,
    }
    state_path = out / "lifecycle.json"
    pod_id = None
    deleted = False
    started = None
    timer = None
    lock = threading.Lock()
    chunks, seen = {}, set()
    expected = digest = None

    def save():
        temporary = state_path.with_suffix(".tmp")
        # Monitor and watchdog can save concurrently; distinct files avoid partial JSON.
        temporary = temporary.with_name(temporary.name + "." + str(threading.get_ident()))
        temporary.write_text(json.dumps(dict(record), indent=2) + "\n")
        temporary.replace(state_path)

    def api(method, path, body=None):
        request = urllib.request.Request(
            "https://api.runpod.io/v2" + path,
            data=json.dumps(body).encode() if body is not None else None, method=method,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
        )
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                data = response.read()
                return json.loads(data) if data else {}
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Runpod HTTP {error.code}; response body withheld") from None

    def entries(response):
        return response if isinstance(response, list) else response.get("pods", [])

    def cleanup():
        nonlocal deleted
        with lock:
            if not pod_id or deleted:
                return
            for _ in range(4):
                try:
                    api("DELETE", "/pods/" + pod_id)
                    deleted = True
                    break
                except RuntimeError as error:
                    if "404" in str(error):
                        deleted = True
                        break
                    record["cleanup_error"] = str(error)
                    save()
                except Exception as error:
                    record["cleanup_error"] = type(error).__name__
                    save()
                time.sleep(2)
            record["delete_request_confirmed"] = deleted
            record["delete_requested_at"] = time.time()
            save()
            print(json.dumps({"stage": "cleanup", "pod_id": pod_id,
                              "delete_request_confirmed": deleted}), flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Lifecycle monitor interrupted; deleting its own pod")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def bundle():
        if expected is None or len(chunks) != expected or not digest:
            return False
        data = base64.b64decode("".join(chunks[i] for i in range(expected)), validate=True)
        if len(data) > 32 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Export checksum mismatch or archive too large")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = archive.getmembers()
            if any((not member.name.startswith("runpod/") and member.name != "runpod")
                   or ".." in Path(member.name).parts or member.issym() or member.islnk()
                   or not (member.isfile() or member.isdir()) for member in members):
                raise RuntimeError("Unsafe artifact archive")
            if sum(member.size for member in members) > 64 * 1024 * 1024:
                raise RuntimeError("Expanded artifact archive is too large")
            archive.extractall(out, filter="data")
        record.update(artifact_bundle_sha256=digest, artifacts_retrieved=True,
                      artifact_bytes=len(data))
        save()
        print(json.dumps({"stage": "artifacts_retrieved", "bytes": len(data),
                          "sha256": digest}), flush=True)
        return True

    command = (
        "mkdir -p /workspace && printf %s " + base64.b64encode(bootstrap).decode()
        + " | base64 -d > /workspace/gpu_vlm_job.sh && printf %s "
        + base64.b64encode(packed).decode()
        + " | base64 -d > /workspace/vlm-input.tar.gz"
        + " && tar -xzf /workspace/vlm-input.tar.gz -C /workspace"
        + " && bash /workspace/gpu_vlm_job.sh"
    )
    body = {
        "name": "robopup-vlm-" + ref[:7] + "-" + secrets.token_hex(3),
        "image": "runpod/pytorch:1.2.0-cu1290-torch290-ubuntu2404", "cloud": "SECURE",
        "gpu": {"id": "NVIDIA GeForce RTX 5090", "count": 1, "minCudaVersion": "12.9"},
        "disk": 30, "entrypoint": ["/bin/bash", "-lc"], "cmd": [command],
        "env": {"ROBOPUP_REF": ref, "PYTHONUNBUFFERED": "1"}, "startSsh": False,
    }
    record.update(image=body["image"], job_script_sha256=hashlib.sha256(bootstrap).hexdigest())
    env = {**os.environ, "RUNPOD_API_KEY": key, "RUNPOD_SKIP_UPDATE_CHECK": "1"}
    try:
        record["pre_existing_pod_count"] = len(entries(api("GET", "/pods")))
        started = time.time()
        record["create_requested_at"] = started
        save()
        try:
            response = api("POST", "/pods", body)
            pod_id = response.get("id")
            if not pod_id:
                raise RuntimeError("Create response missing ID; reconcile by unique name")
        except Exception:
            matches = [pod for pod in entries(api("GET", "/pods"))
                       if pod.get("name") == body["name"]]
            if len(matches) != 1:
                raise
            pod_id = matches[0]["id"]
        record["pod_id"] = pod_id
        save()
        timer = threading.Timer(max(1, TTL - (time.time() - started)), cleanup)
        timer.daemon = True
        timer.start()
        print(json.dumps({"stage": "created", "pod_id": pod_id,
                          "source_ref": ref, "ttl_seconds": TTL}), flush=True)
        done = None
        while time.time() - started < TTL and not deleted:
            pod = api("GET", "/pods/" + pod_id)
            rate = pod.get("cost")
            if type(rate) not in (int, float) or not 0 < rate <= MAX_RATE:
                raise RuntimeError("Actual pod hourly rate is missing or exceeds the bound")
            record["confirmed_gpu_rate_usd_hour"] = rate
            record["pod_status"] = pod.get("status", pod.get("desiredStatus", "unknown"))
            save()
            try:
                result = subprocess.run(
                    ["/workspace/.tools/runpodctl", "pod", "logs", pod_id, "--tail", "5000"],
                    env=env, capture_output=True, text=True, timeout=35,
                )
            except subprocess.TimeoutExpired:
                record["log_poll_timeouts"] = record.get("log_poll_timeouts", 0) + 1
                save()
                continue
            for raw in result.stdout.splitlines():
                try:
                    line = json.loads(raw).get("line", "")
                except (json.JSONDecodeError, AttributeError):
                    continue
                for line in line.splitlines():
                    if line.startswith("ROBOPUP_BUNDLE_SHA256:"):
                        digest = line.split(":", 1)[1]
                    elif line.startswith("ROBOPUP_BUNDLE:"):
                        _, seq, data = line.split(":", 2)
                        index, count = map(int, seq.split("/"))
                        if not 0 <= index < count <= 6000 or len(data) > 6000:
                            raise RuntimeError("Invalid export chunk bounds")
                        chunks[index], expected = data, count
                    elif line.startswith("ROBOPUP_DONE:"):
                        done = int(line.split(":", 1)[1])
                        record["job_exit_code"] = done
                        save()
                    elif line.startswith("ROBOPUP_STAGE:") and line not in seen:
                        print(line.replace(key, "[REDACTED]"), flush=True)
                        seen.add(line)
                    elif any(token in line for token in ("Error", "ERROR", "Traceback", "error:")):
                        if line not in seen:
                            print(line[:400].replace(key, "[REDACTED]"), flush=True)
                            seen.add(line)
            if done is not None and (bundle() or (done != 0 and expected is None)):
                break
            print(json.dumps({"stage": "monitor", "elapsed_seconds": round(time.time() - started),
                              "artifact_chunks": len(chunks)}), flush=True)
            time.sleep(10)
    finally:
        cleanup()
        if timer:
            timer.cancel()
        record["monitor_seconds"] = time.time() - started if started else 0
        record["gpu_cost_estimate_usd"] = (
            record["monitor_seconds"] / 3600 * record.get("confirmed_gpu_rate_usd_hour", MAX_RATE)
        )
        record["cost_is_estimate_not_invoice"] = True
        if pod_id and deleted:
            try:
                record["pod_absent_after_delete"] = not any(
                    pod.get("id") == pod_id for pod in entries(api("GET", "/pods"))
                )
            except Exception as error:
                record["cleanup_verification_error"] = type(error).__name__
        save()
        print(json.dumps({"stage": "finished", "pod_absent": record.get("pod_absent_after_delete"),
                          "estimated_gpu_cost_usd": record["gpu_cost_estimate_usd"],
                          "artifacts_retrieved": record.get("artifacts_retrieved", False)}), flush=True)
    if (not record.get("artifacts_retrieved") or record.get("job_exit_code") != 0
            or not record.get("pod_absent_after_delete")):
        raise RuntimeError("Experiment incomplete; inspect preserved lifecycle record")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.input_dir)
