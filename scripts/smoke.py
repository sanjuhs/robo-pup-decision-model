#!/usr/bin/env python3
"""Functional readiness probe against an already running local arena."""
import json
import urllib.request

from websockets.sync.client import connect


def main():
    with urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=5) as response:
        assert json.load(response)["status"] == "ok"
    with urllib.request.urlopen("http://127.0.0.1:8000/", timeout=5) as response:
        assert "RoboPup Arena" in response.read().decode()
    with connect("ws://127.0.0.1:8000/ws", open_timeout=5) as ws:
        scene = json.loads(ws.recv(timeout=5))
        assert scene["type"] == "scene" and len(scene["geoms"]) > 12
        first = json.loads(ws.recv(timeout=5))
        ws.send(json.dumps({"mode": "approach", "target": [1., 0.]}))
        for _ in range(10):
            second = json.loads(ws.recv(timeout=5))
            if second.get("behavior") == "approach":
                break
        assert second["time"] > first["time"] and second["behavior"] == "approach"
    print("HTTP page + health + live MuJoCo WebSocket control passed")


if __name__ == "__main__":
    main()
