"""Small Vercel HTTP gateway. Physics and model weights are never imported."""

import hmac
import json
import os
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
import httpx

app = FastAPI(title="RoboPup HTTP gateway", version="0.1.0")
MAX_PLAN_BYTES = 2_100_000
MAX_REPLY_BYTES = 262_144


def _service_url(variable):
    # Bindings are runtime values; static Vite builds cannot consume them.
    value = os.environ.get(variable, "").strip().rstrip("/")
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError:
        return None
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        return None
    return value


def _authorize(request):
    token = os.environ.get("ROBOPUP_ARENA_TOKEN", "")
    if not token:
        raise HTTPException(503, "Set a backend workshop access key before enabling this operation")
    provided = request.headers.get("authorization", "")
    if not hmac.compare_digest(provided.encode(), ("Bearer " + token).encode()):
        raise HTTPException(401, "Workshop access key required")


async def _body(request, limit):
    result = bytearray()
    async for chunk in request.stream():
        result.extend(chunk)
        if len(result) > limit:
            raise HTTPException(413, "Request exceeds the size limit")
    return bytes(result)


async def _proxy(url, path, *, method="GET", body=None, headers=None, timeout=15):
    if not url:
        raise HTTPException(503, "Required backend service is not configured")
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            async with client.stream(method, url + path, content=body, headers=headers) as response:
                if response.status_code >= 300:
                    status = response.status_code if response.status_code in (401, 413, 422, 503) else 502
                    raise HTTPException(status, "Backend service rejected or could not complete the request")
                received = bytearray()
                async for chunk in response.aiter_bytes():
                    received.extend(chunk)
                    if len(received) > MAX_REPLY_BYTES:
                        raise HTTPException(502, "Backend response exceeds the size limit")
    except (httpx.HTTPError, TimeoutError):
        raise HTTPException(502, "Backend service connection failed or timed out") from None
    try:
        result = json.loads(received)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(502, "Backend service returned an invalid JSON response") from None
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {
        "status": "ok", "service": "app", "role": "HTTP gateway only",
        "physics": "external MuJoCo backend required",
        "backend_configured": _service_url("ROBOPUP_BACKEND_URL") is not None,
        "planner_binding_configured": _service_url("PLANNER_URL") is not None,
        "simulation_running_here": False, "model_weights_loaded_here": False,
    }


@app.get("/api/config")
def configuration():
    backend = _service_url("ROBOPUP_BACKEND_URL")
    return {
        "backend_url": backend,
        "websocket_url": (backend.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
                          + "/ws") if backend else None,
        "planning_endpoint": "/api/plan", "planning_executes_physics": False,
    }


@app.get("/api/vlm")
async def vlm_status():
    return await _proxy(_service_url("PLANNER_URL"), "/status")


@app.post("/api/plan")
async def plan(request: Request):
    _authorize(request)
    body = await _body(request, MAX_PLAN_BYTES)
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("Expected an object")
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(400, "Expected a JSON planning request") from None
    return await _proxy(
        _service_url("PLANNER_URL"), "/plan", method="POST", body=body,
        headers={"Content-Type": "application/json", "Authorization": request.headers["authorization"]},
        timeout=110,
    )


@app.get("/api/runs")
async def runs(request: Request):
    _authorize(request)
    return await _proxy(_service_url("ROBOPUP_BACKEND_URL"), "/api/runs",
                        headers={"Authorization": request.headers["authorization"]})


@app.post("/api/transcribe")
async def transcribe(request: Request):
    _authorize(request)
    audio = await _body(request, 300_000)
    return await _proxy(
        _service_url("ROBOPUP_BACKEND_URL"), "/api/transcribe", method="POST", body=audio,
        headers={"Content-Type": "audio/wav", "Authorization": request.headers["authorization"]},
    )


@app.get("/ws")
def websocket_location():
    return Response(
        "This gateway does not host simulation sessions. Connect to the configured external backend.",
        status_code=426, headers={"Cache-Control": "no-store"},
    )
