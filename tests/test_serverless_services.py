"""Real HTTP binding transport; fake provider output is not model inference."""

import base64
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from fastapi.testclient import TestClient
from PIL import Image
import pytest
import uvicorn

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gateway = load_module("gateway_service", "services/app/app.py")
planner = load_module("planner_service", "services/planner/app.py")


@contextmanager
def serve(application):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(application, log_level="error", lifespan="off"))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    worker.start()
    deadline = time.monotonic() + 5
    while not server.started and worker.is_alive() and time.monotonic() < deadline:
        time.sleep(.01)
    if not server.started:
        server.should_exit = True
        worker.join(timeout=5)
        listener.close()
        raise AssertionError("HTTP fixture did not start")
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        worker.join(timeout=5)
        listener.close()
        assert not worker.is_alive()


@contextmanager
def provider_fixture(raw_mission):
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append({"path": self.path, "json": json.loads(body)})
            result = {"choices": [{"message": {"content": raw_mission}}]}
            response = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()


def planning_request():
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (120, 180, 50)).save(buffer, format="PNG")
    return {
        "instruction": "Look at the image, then stop",
        "image_png": base64.b64encode(buffer.getvalue()).decode(),
        "energy": .8, "held": False,
    }


def test_app_reads_internal_binding_at_runtime_and_calls_real_http(monkeypatch):
    mission = json.dumps({"summary": "Inspect then stop", "steps": [
        {"action": "look", "object": "none"}, {"action": "stop", "object": "none"},
    ]})
    monkeypatch.setenv("ROBOPUP_ARENA_TOKEN", "workshop-test-value")
    with provider_fixture(mission) as (provider_url, requests), serve(planner.app) as planner_url:
        monkeypatch.setenv("ROBOPUP_VLM_BASE_URL", provider_url)
        monkeypatch.setenv("ROBOPUP_VLM_MODEL", "fixture-vision-model")
        monkeypatch.setenv("PLANNER_URL", planner_url)
        with TestClient(gateway.app) as client:
            assert client.get("/api/health").json()["planner_binding_configured"]
            response = client.post("/api/plan", json=planning_request(),
                                   headers={"Authorization": "Bearer workshop-test-value"})
        assert response.status_code == 200, response.text
        assert response.json()["plan"] == json.loads(mission)
        assert response.json()["inference_verified"] is True
        assert response.json()["physics_executed"] is False
        assert requests[0]["path"] == "/v1/chat/completions"
        image_url = requests[0]["json"]["messages"][0]["content"][0]["image_url"]["url"]
        assert image_url == "data:image/png;base64," + planning_request()["image_png"]
        # The binding URL is read when handling a request, not captured at build/import.
        monkeypatch.delenv("PLANNER_URL")
        with TestClient(gateway.app) as client:
            assert client.get("/api/vlm").status_code == 503


def test_planner_unconfigured_and_bad_image_fail_clearly(monkeypatch):
    monkeypatch.setenv("ROBOPUP_ARENA_TOKEN", "workshop-test-value")
    monkeypatch.delenv("ROBOPUP_VLM_BASE_URL", raising=False)
    headers = {"Authorization": "Bearer workshop-test-value"}
    with TestClient(planner.app) as client:
        assert client.get("/status").json()["configured"] is False
        assert client.post("/plan", json=planning_request(), headers=headers).status_code == 503
        bad = {**planning_request(), "image_png": "x" * 40}
        assert client.post("/plan", json=bad, headers=headers).status_code == 422
        assert client.post("/plan", json=planning_request()).status_code == 401
        assert client.post("/plan", content=b"{broken", headers=headers).status_code == 422
        assert client.post("/plan", json={**planning_request(), "held": "false"},
                           headers=headers).status_code == 422


def test_invalid_remote_plan_is_never_executable(monkeypatch):
    mission = '{"summary":"Unsafe","steps":[{"action":"execute_shell","object":"none"}]}'
    monkeypatch.setenv("ROBOPUP_ARENA_TOKEN", "workshop-test-value")
    with provider_fixture(mission) as (provider_url, _):
        monkeypatch.setenv("ROBOPUP_VLM_BASE_URL", provider_url)
        with TestClient(planner.app) as client:
            response = client.post("/plan", json=planning_request(),
                                   headers={"Authorization": "Bearer workshop-test-value"})
        assert response.status_code == 502
        assert response.json()["detail"] == "Vision-language provider returned an invalid mission"


@pytest.mark.parametrize("path,method", [("/api/plan", "post"), ("/api/runs", "get"),
                                       ("/api/transcribe", "post")])
def test_gateway_requires_access_key_before_forwarding(monkeypatch, path, method):
    monkeypatch.setenv("ROBOPUP_ARENA_TOKEN", "workshop-test-value")
    with TestClient(gateway.app) as client:
        assert getattr(client, method)(path).status_code == 401
    monkeypatch.delenv("ROBOPUP_ARENA_TOKEN")
    with TestClient(gateway.app) as client:
        assert getattr(client, method)(path).status_code == 503


def test_serverless_bundle_reuses_protocol_without_heavy_imports():
    prepared = subprocess.run([sys.executable, "prepare.py"], cwd=ROOT / "services/planner",
                              capture_output=True, text=True)
    assert prepared.returncode == 0, prepared.stderr
    for name in ("vlm_protocol.py", "vlm_runtime.py"):
        assert (ROOT / "services/planner/robopup" / name).read_bytes() == (
            ROOT / "src/robopup" / name
        ).read_bytes()
    check = subprocess.run([
        sys.executable, "-c",
        "import app, sys; assert not {'torch', 'mujoco', 'transformers'} & set(sys.modules)",
    ], cwd=ROOT / "services/planner", capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_services_public_routes_and_internal_binding_are_explicit():
    config = json.loads((ROOT / "vercel.json").read_text())
    assert set(config["services"]) == {"app", "web", "planner"}
    assert config["services"]["app"]["bindings"] == [
        {"type": "service", "service": "planner", "format": "url", "env": "PLANNER_URL"}
    ]
    assert config["rewrites"] == [
        {"source": "/api/:path*", "destination": {"service": "app"}},
        {"source": "/:path*", "destination": {"service": "web"}},
    ]
    assert all(route["destination"]["service"] != "planner" for route in config["rewrites"])
