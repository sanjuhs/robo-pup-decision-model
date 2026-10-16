#!/usr/bin/env python3
"""Run only after explicit root GO: local Vercel services -> real native Qwen."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def token_from_private_file(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) != 0o600:
            raise ValueError("Workshop key must be in a regular mode-0600 file")
        if details.st_uid != os.getuid() or details.st_size > 4096:
            raise ValueError("Workshop key file has unsupported ownership or size")
        value = os.read(descriptor, 4096).decode().strip()
        if not 24 <= len(value) <= 512 or any(char.isspace() for char in value):
            raise ValueError("Workshop key must be bounded non-whitespace text")
        return value
    finally:
        os.close(descriptor)


def request_json(url, *, token=None, payload=None, timeout=10):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = Request(url, json.dumps(payload).encode() if payload is not None else None,
                      headers, method="POST" if payload is not None else "GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(262145)
            if len(body) > 262144:
                raise ValueError("Service response exceeded the inspection limit")
            return response.status, json.loads(body)
    except HTTPError as error:
        error.close()
        raise RuntimeError("Local service returned HTTP " + str(error.code)) from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("Local service connection failed or timed out") from None


def descendants(parent):
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            row = next(row for row in (entry / "status").read_text().splitlines()
                       if row.startswith("PPid:"))
            parents[int(entry.name)] = int(row.split()[1])
        except (OSError, StopIteration, ValueError):
            pass
    owned = {parent}
    while True:
        updated = owned | {pid for pid, ppid in parents.items() if ppid in owned}
        if updated == owned:
            return updated
        owned = updated


def injected_bindings(pids):
    bindings = set()
    for pid in pids:
        try:
            entries = (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        for entry in entries:
            if not entry.startswith(b"PLANNER_URL="):
                continue
            value = entry.split(b"=", 1)[1].decode()
            parsed = urlsplit(value)
            if (parsed.scheme in ("http", "https") and parsed.hostname
                    and not parsed.username and not parsed.password
                    and not parsed.query and not parsed.fragment):
                bindings.add(value)
    return sorted(bindings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/robo-pup-decision-model"))
    parser.add_argument("--key-file", type=Path, default=Path("/tmp/robopup-workshop.token"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=3010)
    parser.add_argument("--run", action="store_true", help="Requires prior explicit root GO")
    args = parser.parse_args()
    if not args.run:
        print("Prepared only. No processes, HTTP requests, or model calls started.")
        return
    args.root = args.root.resolve()
    image_path = args.image or args.root / "benchmarks/vlm-2026-10-16/inputs/eye-start.png"
    key = token_from_private_file(args.key_file)
    image = image_path.read_bytes()
    args.out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.root,
                                     text=True).strip()
    record = {
        "schema": "robopup-real-local-vercel-planner-v1", "source_ref": source,
        "scope": "Local Vercel development services calling authenticated native Qwen; not a cloud deployment",
        "chain": ["app /api/plan", "injected PLANNER_URL internal planner /plan",
                  "native /v1/chat/completions", "actual local Qwen3-VL-2B-Instruct"],
        "cli": "Vercel 64.3.1 dev -L", "cloud_login_used": False,
        "planner_url_manually_set": False, "instruction": "Stop.",
        "image_sha256": hashlib.sha256(image).hexdigest(), "image_file": image_path.name,
        "workshop_key_from_private_file": True, "credentials_in_evidence": False,
        "physics_executed": False, "raw_model_text_available": False,
        "response_note": "The planner validates actual model text and returns parsed plan JSON; full gateway JSON is retained here, not fabricated raw model text.",
        "success": False,
    }
    record["source_file_sha256"] = {
        filename: hashlib.sha256((args.root / filename).read_bytes()).hexdigest()
        for filename in ("services/app/app.py", "services/planner/app.py",
                         "src/robopup/vlm_api.py", "src/robopup/vlm_runtime.py", "vercel.json")
    }
    result_path = args.out / "result.json"
    save(result_path, record)
    process = None
    worker = None
    owned = set()
    log = None
    started = time.monotonic()
    try:
        _, before = request_json("http://127.0.0.1:8000/api/vlm")
        record["native_status_before"] = before
        if not before.get("model_loaded"):
            raise RuntimeError("Native model must be warmed by the browser before this probe")
        env = os.environ.copy()
        for name in ("PLANNER_URL", "RUNPOD_API_KEY", "VERCEL_TOKEN", "HF_TOKEN",
                     "HUGGING_FACE_HUB_TOKEN"):
            env.pop(name, None)
        env.update(ROBOPUP_ARENA_TOKEN=key, ROBOPUP_VLM_API_KEY=key,
                   ROBOPUP_VLM_BASE_URL="http://127.0.0.1:8000/v1",
                   ROBOPUP_VLM_MODEL="Qwen/Qwen3-VL-2B-Instruct",
                   ROBOPUP_BACKEND_URL="http://127.0.0.1:8000",
                   VERCEL_TELEMETRY_DISABLED="1", NO_UPDATE_NOTIFIER="1")
        prep = subprocess.run([str(args.root / ".venv/bin/python"),
                               "services/planner/prepare.py"], cwd=args.root,
                              capture_output=True, text=True, check=True, env=env)
        record["planner_bundle_prepared"] = prep.returncode == 0
        command = ["npm", "exec", "--yes", "--package=vercel@64.3.1", "--", "vercel",
                   "dev", "-L", "--listen", f"127.0.0.1:{args.port}", "--yes"]
        process = subprocess.Popen(command, cwd=args.root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        log_path = args.out / "vercel-local-sanitized.log"
        log = os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w")

        def collect_log():
            for line in process.stdout:
                log.write(line.replace(key, "[REDACTED]"))
                log.flush()

        worker = threading.Thread(target=collect_log, daemon=True)
        worker.start()
        gateway = f"http://127.0.0.1:{args.port}"
        startup_deadline = time.monotonic() + 90
        while time.monotonic() < startup_deadline:
            owned |= descendants(process.pid)
            if process.poll() is not None:
                raise RuntimeError("Owned Vercel process exited during startup")
            try:
                _, health = request_json(gateway + "/api/health", timeout=3)
                if health.get("service") == "app" and health.get("planner_binding_configured"):
                    record["gateway_health"] = health
                    break
            except (RuntimeError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError("Vercel gateway did not expose its injected planner binding")
        owned |= descendants(process.pid)
        record["actual_injected_planner_urls"] = injected_bindings(owned)
        if not record["actual_injected_planner_urls"]:
            raise RuntimeError("Could not independently observe Vercel's injected binding")
        _, status = request_json(gateway + "/api/vlm")
        record["internal_planner_status_via_binding"] = status
        if status.get("service") != "planner" or not status.get("configured"):
            raise RuntimeError("Internal planner is not configured through the actual binding")
        save(result_path, record)
        payload = {"instruction": "Stop.", "image_png": base64.b64encode(image).decode(),
                   "energy": .9, "held": False}
        request_started = time.monotonic()
        record["model_request_started"] = True
        save(result_path, record)
        http_status, response = request_json(gateway + "/api/plan", token=key,
                                             payload=payload, timeout=120)
        record.update(http_status=http_status, gateway_response=response,
                      request_wall_seconds=time.monotonic() - request_started)
        steps = response.get("plan", {}).get("steps")
        record["checks"] = {
            "injected_binding_verified": bool(record["actual_injected_planner_urls"]),
            "actual_model_reported": response.get("model") == "Qwen/Qwen3-VL-2B-Instruct",
            "provider_inference_verified": response.get("inference_verified") is True,
            "no_physics_execution": response.get("physics_executed") is False,
            "expected_stop_plan": steps == [{"action": "stop", "object": "none"}],
            "actual_eye_dimensions": response.get("image", {}).get("width") == 384
                                      and response.get("image", {}).get("height") == 384,
        }
        _, record["native_status_after"] = request_json("http://127.0.0.1:8000/api/vlm")
        record["success"] = all(record["checks"].values())
    except Exception as error:
        record["error_type"] = type(error).__name__
        record["error"] = (str(error).replace(key, "[REDACTED]")
                           if isinstance(error, (RuntimeError, ValueError))
                           else "Probe failed; exception body withheld")
    finally:
        if process is not None:
            owned |= descendants(process.pid)
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=8)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for pid in owned - {process.pid}:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            time.sleep(.5)
            remaining = []
            for pid in owned:
                try:
                    state = (Path("/proc") / str(pid) / "status").read_text()
                    if "State:\tZ" not in state:
                        os.kill(pid, signal.SIGKILL)
                        remaining.append(pid)
                except (FileNotFoundError, ProcessLookupError):
                    pass
            if worker is not None:
                worker.join(timeout=3)
            if log is not None:
                log.close()
            record["owned_vercel_processes_stopped"] = process.poll() is not None
            record["owned_children_requiring_forced_cleanup"] = remaining
        record["wall_seconds"] = time.monotonic() - started
        save(result_path, record)
    print(json.dumps({"output": str(result_path), "success": record["success"],
                      "wall_seconds": record["wall_seconds"]}))
    if not record["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
