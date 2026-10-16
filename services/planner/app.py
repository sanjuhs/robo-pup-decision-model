"""Internal image-to-mission HTTP service; no local weights or simulator imports."""

import base64
import binascii
import hmac
import io
import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from robopup.vlm_protocol import MissionValidationError, parse_mission
from robopup.vlm_runtime import VLMRuntime, VLMRuntimeError

app = FastAPI(title="RoboPup internal vision planner", version="0.1.0")
MAX_REQUEST_BYTES = 2_100_000
MAX_IMAGE_BYTES = 1_500_000


class PlanningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    instruction: str = Field(min_length=1, max_length=2000)
    image_png: str = Field(min_length=20, max_length=2_000_000)
    energy: float = Field(ge=0, le=1, strict=True)
    held: StrictBool


def _runtime():
    # Remote provider only on this lightweight service; never discover/download
    # or import a local Torch checkpoint as a side effect of a web request.
    if not os.environ.get("ROBOPUP_VLM_BASE_URL", "").strip():
        raise HTTPException(503, "A remote vision-language provider must be configured")
    try:
        return VLMRuntime(timeout_s=90)
    except VLMRuntimeError:
        raise HTTPException(503, "Remote vision-language provider configuration is invalid") from None


@app.get("/status")
def status():
    configured = bool(os.environ.get("ROBOPUP_VLM_BASE_URL", "").strip())
    return {
        "service": "planner", "provider": "openai-compatible", "configured": configured,
        "model": os.environ.get("ROBOPUP_VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct"),
        "model_loaded_here": False, "inference_verified": False,
        "detail": "Remote inference is verified per successful request; no local weights are bundled",
    }


@app.post("/plan")
async def plan(request: Request):
    token = os.environ.get("ROBOPUP_ARENA_TOKEN", "")
    if not token:
        raise HTTPException(503, "A workshop access key must be configured before planning")
    if not hmac.compare_digest(request.headers.get("authorization", "").encode(),
                               ("Bearer " + token).encode()):
        raise HTTPException(401, "Workshop access key required")
    received = bytearray()
    async for chunk in request.stream():
        received.extend(chunk)
        if len(received) > MAX_REQUEST_BYTES:
            raise HTTPException(413, "Planning request exceeds the size limit")
    try:
        data = json.loads(received)
        validated = PlanningRequest.model_validate(data)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(422, "Invalid planning request") from None
    try:
        image_bytes = base64.b64decode(validated.image_png, validate=True)
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(413, "Camera image exceeds the size limit")
        with Image.open(io.BytesIO(image_bytes)) as opened:
            if (opened.format != "PNG" or opened.width < 16 or opened.height < 16
                    or opened.width > 1024 or opened.height > 1024):
                raise ValueError("Expected a bounded PNG camera image")
            image = opened.convert("RGB")
    except (binascii.Error, OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(422, "Expected a valid PNG camera image between 16 and 1024 pixels per side") from None
    runtime = _runtime()
    from starlette.concurrency import run_in_threadpool
    try:
        generation = await run_in_threadpool(
            runtime.generate, image, validated.instruction, validated.energy, validated.held
        )
        mission = parse_mission(generation.raw_text)
    except VLMRuntimeError:
        raise HTTPException(502, "Vision-language provider inference failed") from None
    except MissionValidationError:
        raise HTTPException(502, "Vision-language provider returned an invalid mission") from None
    return JSONResponse({
        "plan": mission.as_dict(), "model": generation.model,
        "provider": generation.provider, "timing_ms": generation.timing_ms,
        "image": {"width": image.width, "height": image.height, "source": "request-camera-rgb"},
        "inference_verified": True, "physics_executed": False,
    }, headers={"Cache-Control": "no-store"})
