"""Lifecycle/transport unit tests. Fixtures are NOT evidence of model inference."""

import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import threading

from PIL import Image
import pytest

from robopup.vlm_runtime import VLMRuntime, VLMRuntimeError


@pytest.mark.parametrize("name,value", [
    ("ROBOPUP_VLM_DTYPE", "float16"),
    ("ROBOPUP_VLM_DTYPE", "arbitrary"),
    ("ROBOPUP_VLM_CPU_THREADS", "0"),
    ("ROBOPUP_VLM_CPU_THREADS", "9"),
    ("ROBOPUP_VLM_CPU_THREADS", "3.5"),
])
def test_invalid_local_runtime_configuration_fails_before_loading(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(VLMRuntimeError, match=name):
        VLMRuntime()


def test_explicit_cpu_configuration_is_visible_without_loading(monkeypatch, tmp_path):
    monkeypatch.setenv("ROBOPUP_VLM_DTYPE", "bfloat16")
    monkeypatch.setenv("ROBOPUP_VLM_CPU_THREADS", "4")
    status = VLMRuntime(model_dir=tmp_path).status()
    assert status["dtype_config"] == "bfloat16" and status["cpu_threads_config"] == 4
    assert not status["model_loaded"]


def test_missing_checkpoint_never_claims_a_loaded_model(tmp_path):
    runtime = VLMRuntime(model_dir=tmp_path)
    status = runtime.status()
    assert not status["ready"] and not status["model_loaded"]
    assert status["phase"] == "unavailable"
    assert not list(tmp_path.iterdir())
    with pytest.raises(VLMRuntimeError, match="No local VLM weights"):
        runtime.generate(Image.new("RGB", (32, 24)), "fetch the ball", 0.8, False)
    assert runtime.status()["phase"] == "error"
    assert not list(tmp_path.iterdir())  # No download attempt or generated stub.


def test_provider_configuration_does_not_claim_loaded_or_verified():
    runtime = VLMRuntime(base_url="http://127.0.0.1:7654/v1", api_key="unit-fixture")
    status = runtime.status()
    assert status["ready"] and status["phase"] == "configured"
    assert not status["model_loaded"]
    assert "verified only after a call" in status["detail"]
    assert "unit-fixture" not in json.dumps(status)


@pytest.mark.parametrize("base_url", [
    "ftp://example.invalid/v1", "https://user:password@example.invalid/v1",
    "https://example.invalid/v1?token=value", "https://example.invalid/v1#secret",
])
def test_provider_url_rejects_credential_carriers(base_url):
    with pytest.raises(VLMRuntimeError, match="without credentials"):
        VLMRuntime(base_url=base_url)


def test_invalid_request_does_not_make_provider_call():
    runtime = VLMRuntime(base_url="http://127.0.0.1:7654/v1")
    with pytest.raises(ValueError, match="energy"):
        runtime.generate(Image.new("RGB", (32, 24)), "fetch the ball", 2.0, False)
    with pytest.raises(VLMRuntimeError, match="dimensions"):
        runtime.generate(Image.new("RGB", (1, 1)), "fetch the ball", 0.8, False)
    assert runtime.status()["phase"] == "configured"


def test_real_http_transport_attaches_actual_pixels_but_fixture_is_not_a_model():
    received = []

    class FixtureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            body = json.dumps({"choices": [{"message": {"content": "transport fixture"}}],
                               "usage": {"prompt_tokens": 42, "completion_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), FixtureHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        runtime = VLMRuntime(base_url=f"http://127.0.0.1:{server.server_port}/v1")
        source = Image.new("RGB", (32, 24), (12, 67, 210))
        result = runtime.generate(source, "fetch the ball", 0.8, False)
    finally:
        server.shutdown()
        worker.join(timeout=2)
        server.server_close()
    path, payload = received[0]
    assert path == "/v1/chat/completions"
    content = payload["messages"][0]["content"]
    assert payload["temperature"] == 0 and payload["max_tokens"] == 256
    uri = content[0]["image_url"]["url"]
    decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert decoded.size == source.size and decoded.tobytes() == source.tobytes()
    assert "fetch the ball" in content[1]["text"]
    assert result.raw_text == "transport fixture"
    assert result.metadata["completion_tokens"] == 2
    assert result.timing_ms["generation"] >= 0
    assert runtime.status()["phase"] == "ready" and not runtime.status()["model_loaded"]
