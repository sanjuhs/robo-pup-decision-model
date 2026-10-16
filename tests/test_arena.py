import numpy as np
from fastapi.testclient import TestClient

from robopup.decisions import decide
from robopup.server import app


def test_decisions_stop_recover_approach_push():
    pos = np.array([0., 0., .4])
    args = (pos, 0., 1., [2., 0.], [1.5, 0., .14])
    assert decide("stop", *args).command[0] == 0
    assert decide("approach", *args).command[0] > 0
    assert decide("push", *args).command[0] > 0
    assert decide("follow", pos, 0., .1, [2., 0.], [1., 0., .14]).behavior == "recover"


def test_websocket_real_physics_and_invalid_controls():
    with TestClient(app) as client:
        assert client.get("/api/health").json()["physics"] == "MuJoCo CPU"
        with client.websocket_connect("/ws") as ws:
            assert len(ws.receive_json()["geoms"]) > 12
            first = ws.receive_json()
            ws.send_json({"mode": "approach", "target": [2., 0.]})
            second = ws.receive_json()
            assert second["time"] > first["time"]
            assert second["behavior"] == "approach"
            assert second["position"] != first["position"]
            ws.send_json({"forward": 100})
            # States stream independently of requests; an already queued frame
            # may precede the rejection. Still require the actual validation
            # error within five messages, rather than assuming response ordering.
            for _ in range(5):
                rejection = ws.receive_json()
                if rejection["type"] == "error":
                    break
                assert rejection["type"] == "state"
            assert rejection["type"] == "error"
            assert rejection["message"] == "Invalid controls"


def test_worlds_are_isolated():
    with TestClient(app) as client:
        with client.websocket_connect("/ws") as a, client.websocket_connect("/ws") as b:
            a.receive_json()
            b.receive_json()
            a.receive_json()
            b.receive_json()
            a.send_json({"target": [4., 4.], "mode": "approach"})
            for _ in range(10):
                state = a.receive_json()
                if state.get("target") == [4., 4.]:
                    break
            assert state["target"] == [4., 4.]
            assert b.receive_json()["target"] == [2., 1.]


def test_cross_origin_socket_rejected():
    import pytest
    from starlette.websockets import WebSocketDisconnect
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws", headers={"origin": "https://unrelated.example"}):
                pass
