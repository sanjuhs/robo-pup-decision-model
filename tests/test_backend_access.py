"""Public deployment boundaries; these tests do not contact a model provider."""

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from robopup.server import app, origin_allowed


def test_workshop_key_protects_http_and_socket_but_metadata_is_public(monkeypatch):
    monkeypatch.setenv("ROBOPUP_ARENA_TOKEN", "test-workshop-key")
    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/vlm").status_code == 200
        assert client.get("/api/runs").status_code == 401
        assert client.post("/api/transcribe", content=b"bad").status_code == 401
        assert client.get("/api/runs", headers={"Authorization": "Bearer test-workshop-key"}).status_code == 200
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws") as socket:
                socket.send_json({"type": "auth", "token": "wrong"})
                socket.receive_json()
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws") as socket:
                socket.send_json({"type": "auth", "token": "wrong-🔑"})
                socket.receive_json()
        with client.websocket_connect("/ws") as socket:
            socket.send_json({"type": "auth", "token": "test-workshop-key"})
            assert socket.receive_json()["type"] == "scene"
            assert socket.receive_json()["type"] == "state"


def test_explicit_external_origin_also_requires_workshop_auth(monkeypatch):
    monkeypatch.setenv("ROBOPUP_ALLOWED_ORIGINS", "https://arena.example,https://preview.example")
    monkeypatch.delenv("ROBOPUP_ARENA_TOKEN", raising=False)
    assert not origin_allowed("https://arena.example", "compute.example")
    monkeypatch.setenv("ROBOPUP_ARENA_TOKEN", "test-workshop-key")
    assert origin_allowed("https://arena.example", "compute.example")
    assert not origin_allowed("https://arena.example.attacker.test", "compute.example")
    with TestClient(app) as client:
        with client.websocket_connect("/ws", headers={"origin": "https://arena.example"}) as socket:
            socket.send_json({"type": "auth", "token": "test-workshop-key"})
            assert socket.receive_json()["type"] == "scene"


def test_unavailable_vlm_selection_cannot_silently_execute_a_different_model(monkeypatch):
    from robopup.vlm_runtime import VLMRuntime
    import robopup.server as server

    runtime = VLMRuntime(model_dir="/tmp/robopup-intentionally-absent-checkpoint")
    monkeypatch.setattr(server, "get_vlm_runtime", lambda: runtime)
    import robopup.vlm_runtime as runtime_module
    monkeypatch.setattr(runtime_module, "get_vlm_runtime", lambda: runtime)
    with TestClient(app) as client:
        assert client.get("/api/vlm").json()["available"] is False
        with client.websocket_connect("/ws") as socket:
            socket.receive_json()
            socket.receive_json()
            socket.send_json({"brain": "vlm"})
            for _ in range(8):
                state = socket.receive_json()
                if state["type"] == "error":
                    break
            assert state["type"] == "error"
            assert "unavailable" in state["message"]
            assert socket.receive_json()["brain"] == "neural"
