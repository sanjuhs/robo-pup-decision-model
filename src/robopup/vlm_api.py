"""Private OpenAI-compatible serving of the actual configured local VLM.

Only one user message containing text and an inline PNG is accepted. This route
never downloads a model, fetches an image URL, wraps an already composed prompt,
or falls back to the arena planner. It is a bounded workshop endpoint, not a
multi-tenant inference scheduler. Native mission JSON validation remains the
caller's responsibility; this endpoint returns the real model's raw text.
"""

import asyncio
import base64
import binascii
from concurrent.futures import ThreadPoolExecutor
import hmac
import io
import json
import os
import threading
import time
import uuid

from fastapi import APIRouter, HTTPException, Request
from PIL import Image, UnidentifiedImageError

from robopup.vlm_runtime import get_vlm_runtime

MAX_BODY_BYTES = 2_100_000
MAX_IMAGE_BYTES = 1_500_000
MAX_IMAGE_SIDE = 1024
MAX_PROMPT_CHARS = 16_000
PNG_PREFIX = "data:image/png;base64,"

router = APIRouter()
# A lease is retained until the actual worker finishes, including when its HTTP
# requester disconnects. Cancellation cannot create unbounded background work.
_SERVE_SLOTS = threading.BoundedSemaphore(2)
_SERVE_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="robopup-vlm-api")


def _bad_request(message="Invalid vision chat request"):
    return HTTPException(status_code=400, detail=message)


def _authenticate(request):
    token = os.getenv("ROBOPUP_ARENA_TOKEN", "")
    if not token:
        raise HTTPException(status_code=503, detail="VLM serving requires a workshop access key")
    supplied = request.headers.get("authorization", "")
    if not hmac.compare_digest(supplied.encode("utf-8"), ("Bearer " + token).encode("utf-8")):
        raise HTTPException(status_code=401, detail="Workshop access key required")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate property")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("Non-finite JSON")


async def _read_payload(request):
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError:
            raise _bad_request("Invalid content length") from None
        if length < 0:
            raise _bad_request("Invalid content length")
        if length > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Vision request exceeds the body limit")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="Vision request exceeds the body limit")
        chunks.append(chunk)
    try:
        return json.loads(b"".join(chunks), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise _bad_request("Body must be one valid JSON object") from None


def _decode_png(url):
    if not isinstance(url, str) or not url.startswith(PNG_PREFIX):
        raise _bad_request("Only an inline base64 PNG image is supported")
    encoded = url[len(PNG_PREFIX):]
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise HTTPException(status_code=413, detail="PNG image exceeds the byte limit")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise _bad_request("Invalid base64 PNG image") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="PNG image exceeds the byte limit")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise _bad_request("Image content is not PNG")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size
            if image.format != "PNG" or not 16 <= width <= MAX_IMAGE_SIDE or not 16 <= height <= MAX_IMAGE_SIDE:
                raise _bad_request("PNG dimensions must be between 16 and 1024 on each side")
            if getattr(image, "is_animated", False):
                raise _bad_request("Animated PNG images are not supported")
            image.verify()
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise _bad_request("Invalid PNG image") from None


def _parse_request(payload, runtime):
    if not isinstance(payload, dict) or set(payload) != {"model", "messages", "temperature", "max_tokens"}:
        raise _bad_request("Expected exactly model, messages, temperature and max_tokens")
    if not isinstance(payload["model"], str) or payload["model"] != runtime.model:
        raise _bad_request("Requested model does not match the configured local model")
    if type(payload["temperature"]) not in (int, float) or payload["temperature"] != 0:
        raise _bad_request("Only temperature zero is supported")
    maximum = min(runtime.max_new_tokens, 1024)
    if type(payload["max_tokens"]) is not int or not 1 <= payload["max_tokens"] <= maximum:
        raise _bad_request(f"max_tokens must be between 1 and {maximum}")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) != 1:
        raise _bad_request("Exactly one user message is required")
    message = messages[0]
    if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != "user":
        raise _bad_request("Exactly one user message is required")
    content = message["content"]
    if not isinstance(content, list) or len(content) != 2:
        raise _bad_request("Message must contain one text item and one inline PNG")
    prompt, url = None, None
    for item in content:
        if not isinstance(item, dict):
            raise _bad_request()
        if item.get("type") == "text" and set(item) == {"type", "text"} and prompt is None:
            prompt = item["text"]
        elif item.get("type") == "image_url" and set(item) == {"type", "image_url"} and url is None:
            image_url = item["image_url"]
            if not isinstance(image_url, dict) or set(image_url) != {"url"}:
                raise _bad_request("Image item requires only a URL field")
            url = image_url["url"]
        else:
            raise _bad_request("Message must contain one text item and one inline PNG")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS or "\x00" in prompt:
        raise _bad_request("Prompt must contain 1 to 16000 characters without NUL")
    try:
        prompt.encode("utf-8")
    except UnicodeEncodeError:
        raise _bad_request("Prompt contains invalid Unicode") from None
    return _decode_png(url), prompt, payload["max_tokens"]


def _completion(generation, model):
    if not isinstance(generation.raw_text, str) or not generation.raw_text.strip():
        raise HTTPException(status_code=502, detail="Local VLM returned no text")
    metadata = generation.metadata
    usage = {}
    for source, target in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
        value = metadata.get(source)
        if type(value) is int and value >= 0:
            usage[target] = value
    if "prompt_tokens" in usage and "completion_tokens" in usage:
        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": generation.raw_text},
                     "finish_reason": "length" if metadata.get("truncated") else "stop"}],
        "usage": usage,
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    _authenticate(request)
    payload = await _read_payload(request)
    try:
        runtime = get_vlm_runtime()
        available = runtime.status().get("available")
    except Exception:
        raise HTTPException(status_code=503, detail="Local VLM is unavailable") from None
    if runtime.base_url:
        raise HTTPException(status_code=503, detail="This endpoint requires a local VLM checkpoint")
    if not available:
        raise HTTPException(status_code=503, detail="Local VLM checkpoint or dependencies unavailable")
    image, prompt, maximum = _parse_request(payload, runtime)
    if not _SERVE_SLOTS.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Local VLM is busy; retry later",
                            headers={"Retry-After": "2"})
    try:
        future = _SERVE_POOL.submit(runtime.generate_prompt, image, prompt, max_new_tokens=maximum)
    except Exception:
        _SERVE_SLOTS.release()
        raise HTTPException(status_code=503, detail="Local VLM worker is unavailable") from None
    future.add_done_callback(lambda _finished: _SERVE_SLOTS.release())
    try:
        generation = await asyncio.wrap_future(future)
        return _completion(generation, runtime.model)
    except HTTPException:
        raise
    except Exception:
        # Third-party exceptions can contain signed URLs or credentials. Never
        # return exception bodies, tracebacks, prompts or request headers.
        raise HTTPException(status_code=502, detail="Local VLM inference failed") from None
