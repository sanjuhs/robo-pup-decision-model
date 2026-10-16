#!/usr/bin/env python3
"""Explicitly authorized RTX 5090 experiment; memory-only credential and bounded cleanup.

This creates paid infrastructure. Use only with session authorization. It never
passes the account key into the pod, prints raw API bodies, or deletes other pods.
"""

import argparse
import base64
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TTL = 40 * 60
MAX_RATE = 1.5


def run(out, candidate="v3"):
    out.mkdir(parents=True, exist_ok=False)
    key = os.environ.get("RUNPOD_API_KEY") or getpass.getpass(
        "Runpod credential (hidden; memory only): "
    )
    if not key:
        raise ValueError("Runpod credential is missing")
    ref = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    record = {
        "source_ref": ref,
        "gpu": "RTX 5090",
        "candidate": candidate,
        "ttl_seconds": TTL,
        "total_session_budget_usd": 30,
        "prior_spend_reserve_usd": 1,
        "experiment_allocation_usd": 3,
        "maximum_gpu_rate_usd_hour": MAX_RATE,
        "storage_reserve_usd": 0.2,
        "paid_scope": "One candidate training experiment, no automatic retry/fallback, no network volume",
    }
    state_path = out / "lifecycle.json"
    pod_id = None
    deleted = False
    started = None
    timer = None
    lock = threading.Lock()
    chunks = {}
    expected = None
    digest = None
    seen = set()

    def save():
        state_path.write_text(json.dumps(record, indent=2) + "\n")

    def api(method, path, body=None):
        request = urllib.request.Request(
            "https://api.runpod.io/v2" + path,
            data=json.dumps(body).encode() if body is not None else None,
            method=method,
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
            for attempt in range(4):
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
            print(
                json.dumps(
                    {"stage": "cleanup", "pod_id": pod_id, "delete_request_confirmed": deleted}
                ),
                flush=True,
            )

    def interrupted(signum, frame):
        raise KeyboardInterrupt("Lifecycle monitor interrupted; deleting own pod")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def bundle():
        if expected is None or len(chunks) != expected or not digest:
            return False
        data = base64.b64decode("".join(chunks[i] for i in range(expected)))
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("Export checksum mismatch")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            if any(
                (not m.name.startswith("runpod/") and m.name != "runpod")
                or ".." in Path(m.name).parts
                or m.issym()
                or m.islnk()
                for m in archive.getmembers()
            ):
                raise RuntimeError("Unsafe artifact archive")
            archive.extractall(out, filter="data")
        record["artifact_bundle_sha256"] = digest
        record["artifacts_retrieved"] = True
        record["artifact_bytes"] = len(data)
        save()
        print(
            json.dumps({"stage": "artifacts_retrieved", "bytes": len(data), "sha256": digest}),
            flush=True,
        )
        return True

    bootstrap = (
        ROOT
        / ("scripts/gpu_planner_v4_job.sh" if candidate == "v4" else "scripts/gpu_planner_job.sh")
    ).read_bytes()
    command = (
        "mkdir -p /workspace && printf %s "
        + base64.b64encode(bootstrap).decode()
        + " | base64 -d > /workspace/gpu_planner_job.sh && bash /workspace/gpu_planner_job.sh"
    )
    body = {
        "name": "robopup-planner-" + candidate + "-" + ref[:7] + "-" + secrets.token_hex(3),
        "image": "runpod/pytorch:1.2.0-cu1290-torch290-ubuntu2404",
        "cloud": "SECURE",
        "gpu": {"id": "NVIDIA GeForce RTX 5090", "count": 1, "minCudaVersion": "12.9"},
        "disk": 20,
        "entrypoint": ["/bin/bash", "-lc"],
        "cmd": [command],
        "env": {"ROBOPUP_REF": ref, "PYTHONUNBUFFERED": "1"},
        "startSsh": False,
    }
    record["image"] = body["image"]
    record["job_script_sha256"] = hashlib.sha256(bootstrap).hexdigest()
    env = os.environ.copy()
    env["RUNPOD_API_KEY"] = key
    env["RUNPOD_SKIP_UPDATE_CHECK"] = "1"
    try:
        before = entries(api("GET", "/pods"))
        record["pre_existing_pod_count"] = len(before)
        started = time.time()
        record["create_requested_at"] = started
        save()
        try:
            response = api("POST", "/pods", body)
            pod_id = response.get("id")
            if not pod_id:
                raise RuntimeError("Create response missing pod ID; reconcile by unique name")
        except Exception:
            matches = [p for p in entries(api("GET", "/pods")) if p.get("name") == body["name"]]
            if len(matches) != 1:
                raise
            pod_id = matches[0]["id"]
        record["pod_id"] = pod_id
        save()
        timer = threading.Timer(max(1, TTL - (time.time() - started)), cleanup)
        timer.daemon = True
        timer.start()
        print(
            json.dumps(
                {"stage": "created", "pod_id": pod_id, "source_ref": ref, "ttl_seconds": TTL}
            ),
            flush=True,
        )
        done = None
        while time.time() - started < TTL and not deleted:
            pod = api("GET", "/pods/" + pod_id)
            rate = pod.get("cost")
            if not isinstance(rate, (int, float)) or not 0 < rate <= MAX_RATE:
                raise RuntimeError("Actual pod hourly rate missing or exceeds experiment bound")
            record["confirmed_gpu_rate_usd_hour"] = rate
            record["pod_status"] = pod.get("status", pod.get("desiredStatus", "unknown"))
            save()
            try:
                result = subprocess.run(
                    ["/workspace/.tools/runpodctl", "pod", "logs", pod_id, "--tail", "5000"],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=35,
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
                        index, count = seq.split("/")
                        chunks[int(index)] = data
                        expected = int(count)
                    elif line.startswith("ROBOPUP_DONE:"):
                        done = int(line.split(":", 1)[1])
                        record["job_exit_code"] = done
                        save()
                    elif line.startswith("ROBOPUP_STAGE:") or line.startswith('{"step":'):
                        if line not in seen:
                            print(line.replace(key, "[REDACTED]"), flush=True)
                            seen.add(line)
                    elif (
                        any(x in line for x in ("Error", "ERROR", "Traceback", "error:"))
                        and line not in seen
                    ):
                        print(line[:400].replace(key, "[REDACTED]"), flush=True)
                        seen.add(line)
            if done is not None:
                if bundle():
                    break
                if done != 0 and expected is None:
                    break
            print(
                json.dumps(
                    {
                        "stage": "monitor",
                        "elapsed_seconds": round(time.time() - started),
                        "artifact_chunks": len(chunks),
                    }
                ),
                flush=True,
            )
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
                    p.get("id") == pod_id for p in entries(api("GET", "/pods"))
                )
            except Exception as error:
                record["cleanup_verification_error"] = type(error).__name__
        save()
        print(
            json.dumps(
                {
                    "stage": "finished",
                    "pod_absent": record.get("pod_absent_after_delete"),
                    "estimated_gpu_cost_usd": record["gpu_cost_estimate_usd"],
                    "artifacts_retrieved": record.get("artifacts_retrieved", False),
                }
            ),
            flush=True,
        )
    if (
        not record.get("artifacts_retrieved")
        or record.get("job_exit_code") != 0
        or not record.get("pod_absent_after_delete")
    ):
        raise RuntimeError("Experiment incomplete; inspect preserved lifecycle record")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--candidate", choices=["v3", "v4"], default="v3")
    args = p.parse_args()
    run(args.out, args.candidate)
