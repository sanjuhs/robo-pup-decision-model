"""Opt-in, genuine generative vision-language inference for the mission planner.

The small arena transformer is a different model. This module never substitutes
its answers, a command parser, or canned plans for an unavailable VLM. Model
downloads are an explicit operator action, never a side effect of a web request.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image

from robopup.vlm_protocol import build_planning_prompt

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
DEFAULT_MODEL_DIR = Path("artifacts/vlm/Qwen3-VL-2B-Instruct")
MANIFEST_NAME = "robopup-model-manifest.json"
MAX_REMOTE_RESPONSE_BYTES = 262_144


class _NoProviderRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib can forward Authorization across redirects. Provider endpoints
        # must be configured directly so credentials never follow a redirect.
        return None


class VLMRuntimeError(RuntimeError):
    """A bounded, credential-free runtime failure suitable for the browser."""


@dataclass(frozen=True)
class VLMGeneration:
    raw_text: str
    model: str
    provider: str
    timing_ms: dict
    metadata: dict

    def as_dict(self):
        return asdict(self)


def _positive_int(value, *, name, default, maximum):
    if value is None:
        return default
    try:
        parsed = int(value)
    except (ValueError, TypeError) as exc:
        raise VLMRuntimeError(f"{name} must be an integer") from exc
    if not 1 <= parsed <= maximum:
        raise VLMRuntimeError(f"{name} must be between 1 and {maximum}")
    return parsed


def _rgb_image(image):
    if not isinstance(image, Image.Image):
        try:
            image = Image.fromarray(image)
        except (TypeError, ValueError, AttributeError) as exc:
            raise VLMRuntimeError("The VLM requires a real RGB camera image") from exc
    if image.width < 16 or image.height < 16 or image.width * image.height > 16_777_216:
        raise VLMRuntimeError("Camera image dimensions are outside supported bounds")
    return image.convert("RGB")


def _local_checkpoint_available(directory):
    if not (directory / "config.json").is_file():
        return False
    # Availability is a filesystem check, not a claim that weights loaded or ran.
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        try:
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards or any(not isinstance(shard, str) or Path(shard).name != shard
                                 for shard in shards):
                return False
            return all((directory / shard).is_file() and (directory / shard).stat().st_size > 0
                       for shard in shards)
        except (OSError, ValueError, KeyError, TypeError):
            return False
    weights = [*directory.glob("*.safetensors"), directory / "pytorch_model.bin"]
    return any(path.is_file() and path.stat().st_size > 0 for path in weights)


class VLMRuntime:
    def __init__(self, *, model_dir=None, base_url=None, model=None, api_key=None,
                 max_new_tokens=None, timeout_s=None):
        self.model_dir = Path(model_dir or os.getenv("ROBOPUP_VLM_MODEL_DIR", DEFAULT_MODEL_DIR))
        self.base_url = (base_url or os.getenv("ROBOPUP_VLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.getenv("ROBOPUP_VLM_MODEL", DEFAULT_MODEL)
        self._api_key = api_key if api_key is not None else os.getenv("ROBOPUP_VLM_API_KEY", "")
        self.max_new_tokens = _positive_int(
            max_new_tokens if max_new_tokens is not None else os.getenv("ROBOPUP_VLM_MAX_TOKENS"),
            name="ROBOPUP_VLM_MAX_TOKENS", default=256, maximum=1024,
        )
        self.timeout_s = _positive_int(
            timeout_s if timeout_s is not None else os.getenv("ROBOPUP_VLM_TIMEOUT_S"),
            name="ROBOPUP_VLM_TIMEOUT_S", default=90, maximum=300,
        )
        self.dtype_config = os.getenv("ROBOPUP_VLM_DTYPE")
        if self.dtype_config not in (None, "float32", "bfloat16", "auto"):
            raise VLMRuntimeError("ROBOPUP_VLM_DTYPE must be float32, bfloat16, or auto")
        self.cpu_threads = _positive_int(
            os.getenv("ROBOPUP_VLM_CPU_THREADS"), name="ROBOPUP_VLM_CPU_THREADS",
            default=2, maximum=8,
        )
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if (parsed.scheme not in ("http", "https") or not parsed.hostname
                    or parsed.username or parsed.password or parsed.query or parsed.fragment):
                raise VLMRuntimeError("VLM base URL must be an HTTP(S) URL without credentials")
        self._model = None
        self._processor = None
        self._inference_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._phase = "not_loaded"
        self._last_error = None
        self._revision = None
        manifest = self.model_dir / MANIFEST_NAME
        if not self.base_url and manifest.is_file():
            try:
                data = json.loads(manifest.read_text())
                self._revision = data.get("revision")
                if model is None and not os.getenv("ROBOPUP_VLM_MODEL"):
                    self.model = data.get("model_id", self.model)
            except (OSError, ValueError, TypeError):
                # Corrupt metadata must not imply successful inference/provenance.
                self._last_error = "Local model manifest could not be read"

    def status(self):
        remote = bool(self.base_url)
        checkpoint = _local_checkpoint_available(self.model_dir) if not remote else False
        dependencies = all(importlib.util.find_spec(name) is not None for name in (
            "transformers", "accelerate", "torch",
        )) if not remote else True
        with self._state_lock:
            phase, error = self._phase, self._last_error
            loaded = self._model is not None
        available = remote or (checkpoint and dependencies)
        if phase == "not_loaded":
            phase = "configured" if remote else ("available" if available else "unavailable")
        if remote:
            detail = "OpenAI-compatible provider configured; inference is verified only after a call"
        elif not checkpoint:
            detail = "No local VLM weights; run scripts/download_vlm.py explicitly"
        elif not dependencies:
            detail = "Local weights found; install the optional vlm dependencies"
        elif not loaded:
            detail = "Local checkpoint available; it has not been loaded or tested by this process"
        else:
            detail = "Local VLM loaded; mission JSON is validated separately before execution"
        return {
            "ready": available, "phase": phase, "model": self.model,
            "available": available, "status": phase, "reason": detail,
            "provider": "openai-compatible" if remote else "transformers-local",
            "model_loaded": loaded, "checkpoint_available": checkpoint,
            "dependencies_available": dependencies, "revision": self._revision,
            "detail": detail, "last_error": error,
            "dtype_config": self.dtype_config or "default",
            "cpu_threads_config": self.cpu_threads,
            "max_generation_seconds": self.timeout_s,
        }

    def _set_phase(self, phase, error=None):
        with self._state_lock:
            self._phase, self._last_error = phase, error

    def generate(self, image, instruction, energy, held_object):
        """Run the actual model. Return raw text; this does not authorize execution."""
        prompt = build_planning_prompt(instruction, energy=energy, held=held_object)
        rgb = _rgb_image(image)
        started = time.perf_counter()
        # One process-wide model can serve sessions without concurrent GPU spikes.
        with self._inference_lock:
            queue_ms = (time.perf_counter() - started) * 1000
            try:
                if self.base_url:
                    result = self._generate_remote(rgb, prompt)
                else:
                    result = self._generate_local(rgb, prompt)
            except VLMRuntimeError as exc:
                self._set_phase("error", str(exc))
                raise
            except Exception as exc:
                # Third-party exceptions can contain endpoint/token-bearing URLs.
                message = "VLM inference failed; inspect the local runtime configuration"
                self._set_phase("error", message)
                raise VLMRuntimeError(message) from exc
            self._set_phase("ready")
        return VLMGeneration(
            raw_text=result[0], model=self.model,
            provider="openai-compatible" if self.base_url else "transformers-local",
            timing_ms={**result[1], "queue": queue_ms,
                       "total": (time.perf_counter() - started) * 1000},
            metadata={**result[2], "revision": self._revision,
                      "image_width": rgb.width, "image_height": rgb.height,
                      "max_new_tokens": self.max_new_tokens},
        )

    def _generate_local(self, rgb, prompt):
        if not self.status()["ready"]:
            raise VLMRuntimeError(self.status()["detail"])
        load_ms = 0.0
        if self._model is None:
            self._set_phase("loading")
            started = time.perf_counter()
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
                cuda = torch.cuda.is_available()
                if not cuda:
                    torch.set_num_threads(self.cpu_threads)
                dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                         "auto": "auto"}.get(self.dtype_config)
                if dtype is None:
                    dtype = torch.bfloat16 if cuda else torch.float32
                self._processor = AutoProcessor.from_pretrained(
                    self.model_dir, local_files_only=True, trust_remote_code=False,
                    max_pixels=262_144,
                )
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self.model_dir, local_files_only=True, trust_remote_code=False,
                    torch_dtype=dtype,
                    device_map="auto",
                ).eval()
            except Exception as exc:
                self._processor, self._model = None, None
                raise VLMRuntimeError("Could not load the local VLM checkpoint") from exc
            load_ms = (time.perf_counter() - started) * 1000
        import torch
        self._set_phase("generating")
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": rgb}, {"type": "text", "text": prompt},
        ]}]
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(self._model.device)
        input_tokens = int(inputs["input_ids"].shape[1])
        decoding_started = time.perf_counter()
        with torch.inference_mode():
            output = self._model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True,
                max_time=self.timeout_s,
            )
        decoding_seconds = time.perf_counter() - decoding_started
        continuation = output[:, input_tokens:]
        raw = self._processor.batch_decode(
            continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        if not raw:
            raise VLMRuntimeError("The VLM produced no text")
        output_tokens = int(continuation.shape[1])
        eos = self._model.generation_config.eos_token_id
        eos = eos if isinstance(eos, (tuple, list)) else [eos]
        ended_with_eos = output_tokens > 0 and int(continuation[0, -1]) in eos
        token_limit = output_tokens >= self.max_new_tokens
        time_limit = decoding_seconds >= self.timeout_s
        return raw, {"load": load_ms, "generation": (time.perf_counter() - started) * 1000}, {
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "device": str(self._model.device), "deterministic_decoding": True,
            "parameter_count": sum(parameter.numel() for parameter in self._model.parameters()),
            "dtype": str(next(self._model.parameters()).dtype),
            "dtype_config": self.dtype_config or "default",
            "cpu_threads_config": self.cpu_threads,
            "cpu_threads": torch.get_num_threads() if self._model.device.type == "cpu" else None,
            "max_generation_seconds": self.timeout_s,
            "timeout_is_cooperative": True,
            "token_limit_reached": token_limit, "time_limit_reached": time_limit,
            "truncated": (token_limit or time_limit) and not ended_with_eos,
            "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated()
                                          if torch.cuda.is_available() else None),
        }

    def _generate_remote(self, rgb, prompt):
        self._set_phase("generating")
        image_bytes = io.BytesIO()
        rgb.save(image_bytes, format="PNG")
        uri = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode("ascii")
        payload = {"model": self.model, "temperature": 0, "max_tokens": self.max_new_tokens,
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": uri}},
                       {"type": "text", "text": prompt},
                   ]}]}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = "Bearer " + self._api_key
        endpoint = self.base_url + "/chat/completions"
        request = Request(endpoint, json.dumps(payload).encode(), headers, method="POST")
        started = time.perf_counter()
        try:
            with build_opener(_NoProviderRedirects()).open(request, timeout=self.timeout_s) as response:
                body = response.read(MAX_REMOTE_RESPONSE_BYTES + 1)
        except HTTPError as exc:
            exc.close()
            raise VLMRuntimeError(f"VLM provider returned HTTP {exc.code}") from None
        except (URLError, TimeoutError, OSError):
            raise VLMRuntimeError("VLM provider connection failed or timed out") from None
        if len(body) > MAX_REMOTE_RESPONSE_BYTES:
            raise VLMRuntimeError("VLM provider response exceeded the size limit")
        try:
            data = json.loads(body)
            raw = data["choices"][0]["message"]["content"]
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Missing text")
            usage = data.get("usage", {})
            # Only propagate known numeric counters; never arbitrary provider metadata.
            tokens = {name: usage[name] for name in ("prompt_tokens", "completion_tokens")
                      if isinstance(usage, dict) and type(usage.get(name)) is int}
        except (ValueError, KeyError, IndexError, TypeError):
            raise VLMRuntimeError("VLM provider returned an invalid chat response") from None
        return raw.strip(), {"generation": (time.perf_counter() - started) * 1000, "load": 0.0}, {
            **tokens, "deterministic_decoding_requested": True,
        }


@lru_cache(maxsize=1)
def get_vlm_runtime():
    """Shared lazy runtime. Environment configuration is read once per process."""
    return VLMRuntime()
