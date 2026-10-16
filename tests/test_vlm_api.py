"""HTTP serving contracts with a controlled adapter, not real-model evidence."""

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import io
import json
import threading

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from robopup import vlm_api
from robopup.vlm_runtime import VLMGeneration

FIXTURE_TOKEN = "private-workshop-api-fixture"
RAW_FIXTURE = '{"summary":"Contract fixture only.","steps":[{"action":"stop","object":"none"}]}'


class ControlledLocalAdapter:
    """Records host calls without loading weights or claiming LLM competence."""

    model = "controlled-local-vlm"
    max_new_tokens = 128
    base_url = ""
    available = True

    def __init__(self, *, gate=None, output=RAW_FIXTURE, metadata=None):
        self.calls = []
        self.gate, self.output = gate, output
        self.metadata = {"input_tokens": 20, "output_tokens": 9} if metadata is None else metadata
        self.condition = threading.Condition()

    def status(self):
        return {"available": self.available}

    def generate_prompt(self, image, prompt, *, max_new_tokens):
        with self.condition:
            self.calls.append({"image": image.copy(), "prompt": prompt, "max_new_tokens": max_new_tokens})
            self.condition.notify_all()
        if self.gate is not None and not self.gate.wait(timeout=5):
            raise RuntimeError("Controlled adapter gate timed out")
        if isinstance(self.output, Exception):
            raise self.output
        return VLMGeneration(raw_text=self.output, model=self.model, provider="controlled-contract-adapter",
                             timing_ms={"total": 1.0}, metadata=self.metadata)


def png_url(*, size=(32, 24), format="PNG"):
    image = Image.new("RGB", size, (14, 78, 193))
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return vlm_api.PNG_PREFIX + base64.b64encode(buffer.getvalue()).decode("ascii")


def payload():
    return {"model": "controlled-local-vlm", "temperature": 0, "max_tokens": 64,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": png_url()}},
                {"type": "text", "text": "Already composed vision prompt: inspect this image."},
            ]}]}


@contextmanager
def serving_client(monkeypatch, adapter=None):
    adapter = adapter or ControlledLocalAdapter()
    monkeypatch.setenv("ROBOPUP_ARENA_TOKEN", FIXTURE_TOKEN)
    monkeypatch.setattr(vlm_api, "get_vlm_runtime", lambda: adapter)
    app = FastAPI()
    app.include_router(vlm_api.router)
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer " + FIXTURE_TOKEN
        yield client, adapter


def test_private_http_endpoint_preserves_actual_png_and_composed_prompt_without_double_wrapping(monkeypatch):
    with serving_client(monkeypatch) as (client, adapter):
        request = payload()
        result = client.post("/v1/chat/completions", json=request)
        assert result.status_code == 200, result.text
        body = result.json()
        assert body["object"] == "chat.completion" and body["model"] == adapter.model
        assert body["id"].startswith("chatcmpl-") and type(body["created"]) is int
        assert body["choices"] == [{"index": 0, "message": {"role": "assistant", "content": RAW_FIXTURE},
                                    "finish_reason": "stop"}]
        assert body["usage"] == {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29}
        assert len(adapter.calls) == 1
        call = adapter.calls[0]
        assert call["prompt"] == request["messages"][0]["content"][1]["text"]
        assert call["max_new_tokens"] == 64
        assert call["image"].mode == "RGB" and call["image"].size == (32, 24)
        np.testing.assert_array_equal(np.asarray(call["image"]), np.full((24, 32, 3), [14, 78, 193], dtype=np.uint8))
        assert "controlled-contract-adapter" not in result.text
        assert FIXTURE_TOKEN not in result.text


def test_reversed_content_order_is_supported_without_changing_prompt(monkeypatch):
    request = payload()
    request["messages"][0]["content"].reverse()
    with serving_client(monkeypatch) as (client, adapter):
        assert client.post("/v1/chat/completions", json=request).status_code == 200
        assert len(adapter.calls) == 1


def test_missing_secret_binding_disables_endpoint_even_on_localhost(monkeypatch):
    with serving_client(monkeypatch) as (client, adapter):
        monkeypatch.delenv("ROBOPUP_ARENA_TOKEN")
        result = client.post("/v1/chat/completions", json=payload())
        assert result.status_code == 503
        assert result.json()["detail"] == "VLM serving requires a workshop access key"
        assert adapter.calls == []


@pytest.mark.parametrize("authorization", ["", "Bearer wrong", "Basic private-workshop-api-fixture"])
def test_missing_or_wrong_authentication_does_not_invoke_model(monkeypatch, authorization):
    with serving_client(monkeypatch) as (client, adapter):
        response = client.post("/v1/chat/completions", json=payload(), headers={"Authorization": authorization})
        assert response.status_code == 401
        assert adapter.calls == []


def test_non_ascii_header_and_configured_secret_never_cause_hmac_typeerror(monkeypatch):
    with serving_client(monkeypatch) as (client, adapter):
        response = client.post("/v1/chat/completions", json=payload(),
                               headers=[(b"Authorization", b"Bearer wrong-\xff")])
        assert response.status_code == 401
        monkeypatch.setenv("ROBOPUP_ARENA_TOKEN", "unicode-key-\N{DOG FACE}")
        assert client.post("/v1/chat/completions", json=payload()).status_code == 401
        assert adapter.calls == []


@pytest.mark.parametrize("change", [
    {"model": "some-other-model"}, {"model": None}, {"model": ["controlled-local-vlm"]},
    {"temperature": 0.5}, {"temperature": False}, {"temperature": "0"},
    {"max_tokens": 0}, {"max_tokens": -1}, {"max_tokens": 129}, {"max_tokens": 1025},
    {"max_tokens": 1.5}, {"max_tokens": True}, {"max_tokens": "64"},
    {"stream": True}, {"messages": []}, {"messages": "user"},
    {"messages": [{"role": "system", "content": []}]},
])
def test_invalid_top_level_contract_never_invokes_model(monkeypatch, change):
    with serving_client(monkeypatch) as (client, adapter):
        request = payload()
        request.update(change)
        assert client.post("/v1/chat/completions", json=request).status_code == 400
        assert adapter.calls == []


@pytest.mark.parametrize("content", [
    [], [{"type": "text", "text": "only text"}],
    [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
    [{"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}},
     {"type": "text", "text": "inspect"}],
    [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
     {"type": "text", "text": "inspect"}],
    [{"type": "image_url", "image_url": {"url": png_url(), "detail": "high"}},
     {"type": "text", "text": "inspect"}],
    [{"type": "image_url", "image_url": {"url": png_url()}}, {"type": "text", "text": ""}],
    [{"type": "image_url", "image_url": {"url": png_url()}}, {"type": "text", "text": 1}],
    [{"type": "image_url", "image_url": {"url": png_url()}}, {"type": "text", "text": "x" * 16_001}],
    [{"type": "image_url", "image_url": {"url": png_url()}}, {"type": "text", "text": "nul\0byte"}],
])
def test_unknown_items_external_image_urls_and_unbounded_prompts_are_rejected(monkeypatch, content):
    with serving_client(monkeypatch) as (client, adapter):
        request = payload()
        request["messages"][0]["content"] = content
        assert client.post("/v1/chat/completions", json=request).status_code == 400
        assert adapter.calls == []


@pytest.mark.parametrize("url", [
    vlm_api.PNG_PREFIX + "not base64!",
    vlm_api.PNG_PREFIX + base64.b64encode(b"not PNG content").decode(),
    png_url(format="JPEG"), png_url(size=(1025, 16)), png_url(size=(16, 1025)), png_url(size=(8, 8)),
    vlm_api.PNG_PREFIX + base64.b64encode(b"\x89PNG\r\n\x1a\ntruncated").decode(),
])
def test_invalid_png_payloads_fail_before_model_emissions(monkeypatch, url):
    with serving_client(monkeypatch) as (client, adapter):
        request = payload()
        request["messages"][0]["content"][0]["image_url"]["url"] = url
        assert client.post("/v1/chat/completions", json=request).status_code == 400
        assert adapter.calls == []


def test_body_and_decoded_image_limits_are_independent_and_reject_before_inference(monkeypatch):
    with serving_client(monkeypatch) as (client, adapter):
        large = " " * (vlm_api.MAX_BODY_BYTES + 1)
        assert client.post("/v1/chat/completions", content=large,
                           headers={"Content-Type": "application/json"}).status_code == 413
        request = payload()
        request["messages"][0]["content"][0]["image_url"]["url"] = vlm_api.PNG_PREFIX + base64.b64encode(
            b"x" * (vlm_api.MAX_IMAGE_BYTES + 1)).decode()
        assert client.post("/v1/chat/completions", json=request).status_code == 413
        assert adapter.calls == []


def test_chunked_body_size_is_checked_without_trusting_content_length(monkeypatch):
    with serving_client(monkeypatch) as (client, adapter):
        chunks = iter([b" " * 1_100_000, b" " * 1_100_000])
        response = client.post("/v1/chat/completions", content=chunks,
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 413
        assert adapter.calls == []


@pytest.mark.parametrize("body", [
    b"not JSON", b"[]", b"null", b"\xff", b"{" * 1100,
    b'{"model":"x","model":"y","messages":[],"temperature":0,"max_tokens":1}',
    b'{"model":"x","messages":[],"temperature":NaN,"max_tokens":1}',
])
def test_malformed_json_and_duplicate_keys_are_rejected(monkeypatch, body):
    with serving_client(monkeypatch) as (client, adapter):
        assert client.post("/v1/chat/completions", content=body,
                           headers={"Content-Type": "application/json"}).status_code == 400
        assert adapter.calls == []


def test_non_json_body_is_rejected_without_inference(monkeypatch):
    with serving_client(monkeypatch) as (client, adapter):
        assert client.post("/v1/chat/completions", content="{}").status_code == 415
        assert adapter.calls == []


@pytest.mark.parametrize("remote", [False, True])
def test_missing_local_checkpoint_and_remote_recursion_configuration_are_not_served(monkeypatch, remote):
    adapter = ControlledLocalAdapter()
    adapter.base_url = "https://provider.invalid/v1" if remote else ""
    adapter.available = remote
    with serving_client(monkeypatch, adapter) as (client, adapter):
        response = client.post("/v1/chat/completions", json=payload())
        assert response.status_code == 503
        assert adapter.calls == []


def test_model_failure_is_sanitized_and_releases_work_slot(monkeypatch):
    adapter = ControlledLocalAdapter(output=RuntimeError("private-provider-token-bearing-diagnostic"))
    with serving_client(monkeypatch, adapter) as (client, adapter):
        for _ in range(3):
            response = client.post("/v1/chat/completions", json=payload())
            assert response.status_code == 502
            assert response.json()["detail"] == "Local VLM inference failed"
            assert "private-provider-token-bearing-diagnostic" not in response.text
        assert len(adapter.calls) == 3


def test_truncation_and_usage_are_reported_without_arbitrary_adapter_metadata(monkeypatch):
    adapter = ControlledLocalAdapter(metadata={"input_tokens": 20, "output_tokens": 64,
                                               "truncated": True, "private": "not-public"})
    with serving_client(monkeypatch, adapter) as (client, _adapter):
        response = client.post("/v1/chat/completions", json=payload())
        assert response.status_code == 200
        assert response.json()["choices"][0]["finish_reason"] == "length"
        assert response.json()["usage"]["total_tokens"] == 84
        assert "not-public" not in response.text


def test_unknown_token_counts_are_not_fabricated(monkeypatch):
    adapter = ControlledLocalAdapter(metadata={"input_tokens": True, "output_tokens": "nine"})
    with serving_client(monkeypatch, adapter) as (client, _adapter):
        assert client.post("/v1/chat/completions", json=payload()).json()["usage"] == {}


def test_two_inflight_calls_are_bounded_and_third_is_rejected_without_inference(monkeypatch):
    gate = threading.Event()
    adapter = ControlledLocalAdapter(gate=gate)
    with serving_client(monkeypatch, adapter) as (client, adapter):
        with ThreadPoolExecutor(max_workers=2) as callers:
            first = callers.submit(client.post, "/v1/chat/completions", json=copy.deepcopy(payload()))
            second = callers.submit(client.post, "/v1/chat/completions", json=copy.deepcopy(payload()))
            try:
                with adapter.condition:
                    assert adapter.condition.wait_for(lambda: len(adapter.calls) == 2, timeout=3)
                response = client.post("/v1/chat/completions", json=payload())
                assert response.status_code == 429
                assert response.headers["Retry-After"] == "2"
                assert len(adapter.calls) == 2
            finally:
                gate.set()
            assert first.result(timeout=3).status_code == 200
            assert second.result(timeout=3).status_code == 200
        assert client.post("/v1/chat/completions", json=payload()).status_code == 200
        assert len(adapter.calls) == 3


def test_precomposed_structured_prompt_is_forwarded_without_instruction_wrapping(monkeypatch):
    """A JSON-shaped prompt remains task data, not a second protocol wrapper."""
    with serving_client(monkeypatch) as (client, adapter):
        request = payload()
        request["messages"][0]["content"][1]["text"] = json.dumps({"already": "structured model prompt"})
        assert client.post("/v1/chat/completions", json=request).status_code == 200
        assert adapter.calls[0]["prompt"] == '{"already": "structured model prompt"}'
