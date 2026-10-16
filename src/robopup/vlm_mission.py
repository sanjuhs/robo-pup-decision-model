"""Nonblocking, sequential generative missions over evaluated physical skills.

The VLM sees the onboard perspective RGB. Existing v4 skills retain their
overhead RGB grounding. No model output directly controls motors or executes code.
"""

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import time

import numpy as np
from PIL import Image

from robopup.brain import BrainSession
from robopup.eye_grounding import EyeGrounding, require_visible_target
from robopup.vision import EyeCamera
from robopup.vlm_protocol import parse_mission, step_execution, step_status

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robopup-vlm")


class VLMMission:
    def __init__(self, env, runtime):
        self.env, self.runtime = env, runtime
        self.skills = BrainSession(env)
        self.eye = None
        self.pending = None
        self.instruction = ""
        self.phase, self.reason = "idle", "Ready for a generative vision-language mission"
        self.latest_png, self.raw_response = None, ""
        self.plan = None
        self.steps, self.current_step = [], 0
        self.history = []
        self.complete = self.failed = False
        self.started = self.step_started = 0.0
        self.inference_ms = 0.0
        self.closed = False
        self.replans = 0
        self.expected_remaining = None
        self.execution_s = 0.0
        self.last_tick_time = 0.0
        self.wall_started = 0.0
        self.grounding = None
        self.eye_grounding = None

    @property
    def executor(self):
        return self.skills.executor

    def begin(self, instruction):
        if self.pending is not None and not self.pending.done():
            raise ValueError("A VLM request is still running; wait before sending another mission")
        status = self.runtime.status()
        if not status.get("available"):
            raise ValueError(status.get("reason", "Generative VLM is not configured"))
        self.instruction = instruction
        self.complete = self.failed = False
        self.steps, self.current_step, self.history = [], 0, []
        self.plan = None
        self.eye_grounding = None
        self.started = self.env.data.time
        self.execution_s = 0.0
        self.last_tick_time = self.started
        self.wall_started = time.monotonic()
        self.replans = 0
        self._request_plan()

    def _request_plan(self):
        if self.eye is None:
            self.eye = EyeCamera(self.env)
        image = self.eye.capture()
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="PNG")
        self.latest_png = base64.b64encode(buffer.getvalue()).decode()
        self.phase, self.reason = "thinking", "VLM is inspecting the actual robot-eye image"
        prompt = self.instruction
        completed = [s.as_dict() for s in self.steps[:self.current_step]]
        self.expected_remaining = None
        if completed:
            remaining = tuple(self.steps[self.current_step:])
            self.expected_remaining = remaining
            prompt = (
                "Your remaining mission is: "
                + ", then ".join(step_execution(s).command for s in remaining)
                + ". Verified completed steps: " + json.dumps(completed)
                + ". Those completed tasks are finished; do not include them again. "
                "Inspect the fresh image and return only the remaining tasks in order. "
                "You may first rest, look or turn if necessary."
            )
        self.observation_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
        energy, held = float(self.env.energy), bool(self.env.held)
        self.pending = asyncio.get_running_loop().run_in_executor(
            _POOL,
            lambda: self.runtime.generate(
                image, prompt, energy=energy, held_object=held
            ),
        )
        self.requested_at = time.monotonic()

    def _activate(self):
        step = self.steps[self.current_step]
        # A generative plan is not evidence that its target is in view. Recheck
        # actual RGB after inference, without object positions or simulator IDs.
        if self.grounding is None:
            self.grounding = EyeGrounding()
        activation_image = self.eye.capture()
        self.eye_grounding = {
            **self.grounding.observe(activation_image),
            "eye_rgb_sha256": hashlib.sha256(activation_image.tobytes()).hexdigest(),
            "checked_at_sim_s": float(self.env.data.time),
        }
        try:
            require_visible_target(step, self.eye_grounding, held=bool(self.env.held))
        except ValueError as exc:
            self.failed, self.phase, self.reason = True, "error", str(exc)
            self.env.grip(False)
            return
        self.step_started = self.env.data.time
        execution = step_execution(step)
        if execution.kind == "v4":
            self.skills.begin(execution.command)
        self.phase = "executing"
        self.reason = f"Step {self.current_step + 1}: {step.action} {step.object}"

    def tick(self):
        if self.closed or self.failed or self.complete or self.phase == "idle":
            return np.zeros(2), False
        elapsed = max(0.0, self.env.data.time - self.last_tick_time)
        self.last_tick_time = self.env.data.time
        if self.phase == "executing":
            self.execution_s += elapsed
        if self.execution_s > 360 or time.monotonic() - self.wall_started > 1800:
            self.failed, self.phase = True, "error"
            self.reason = "Mission execution timed out" if self.execution_s > 360 else "Mission wall time exceeded 30 minutes"
            self.env.grip(False)
            return np.zeros(2), False
        if self.env.upright < 0.45 or self.env.data.qpos[2] < 0.19:
            self.failed, self.phase, self.reason = True, "error", "Robot fell; reset to continue"
            return np.zeros(2), False
        if self.pending is not None:
            if not self.pending.done():
                return np.zeros(2), False
            future, self.pending = self.pending, None
            try:
                result = future.result()
                self.raw_response = result.raw_text
                self.inference_ms = result.timing_ms["total"]
                plan = parse_mission(result.raw_text)
                self.history.append({
                    "eye_rgb_sha256": self.observation_sha256,
                    "raw_response": self.raw_response,
                    "plan": plan.as_dict(),
                    "timing_ms": self.inference_ms,
                    "metadata": result.metadata,
                })
                self.plan = plan
                if self.expected_remaining is not None:
                    remaining = list(plan.steps)
                    auxiliaries = {"rest", "look", "turn_left", "turn_right"}
                    while remaining and remaining[0].action in auxiliaries and remaining[0] != self.expected_remaining[0]:
                        remaining.pop(0)
                    if tuple(remaining) != self.expected_remaining:
                        raise ValueError("Replan changed the remembered remaining task sequence")
                self.steps = self.steps[:self.current_step] + list(plan.steps)
                if len(self.steps) > 16:
                    raise ValueError("Mission exceeded its total step limit")
                self._activate()
                if self.failed:
                    return np.zeros(2), False
            except Exception:
                # Runtime exceptions can contain signed URLs or credentials.
                # Diagnostics retain raw model text, never exception bodies.
                self.failed, self.phase = True, "error"
                self.reason = "VLM request or structured plan validation failed; no new action executed"
                return np.zeros(2), False
        if self.plan is None:
            return np.zeros(2), False
        step = self.steps[self.current_step]
        execution = step_execution(step)
        if execution.kind == "direct":
            self.env.motion, self.env.turn_gain = "trot", 0.5
            command = np.array([0.12, 0.7 if step.action == "turn_left" else -0.7])
            if step.action == "look":
                command = np.zeros(2)
            learned_motor = False
            complete, failed = False, False
        else:
            command, learned_motor = self.skills.tick()
            complete, failed = bool(self.executor.complete), bool(self.executor.failed)
        outcome = step_status(
            step, elapsed_sim_s=float(self.env.data.time - self.step_started),
            executor_complete=complete, executor_failed=failed,
        )
        if outcome == "failed":
            self.failed, self.phase = True, "error"
            self.reason = f"Step {self.current_step + 1} failed: {self.executor.reason}"
            self.env.grip(False)
            return np.zeros(2), False
        if outcome == "complete":
            self.current_step += 1
            if self.current_step == len(self.steps):
                self.complete, self.phase = True, "complete"
                self.reason = "Mission complete: all physical or bounded-duration steps verified"
            elif self.replans < 7:
                self.replans += 1
                self._request_plan()
            else:
                self.failed, self.phase, self.reason = True, "error", "VLM replan limit reached"
            return np.zeros(2), False
        return command, learned_motor

    def status(self):
        configured = self.runtime.status()
        rows = []
        for i, step in enumerate(self.steps):
            state = "complete" if i < self.current_step else "pending"
            if i == self.current_step:
                state = "failed" if self.failed else "running" if self.phase == "executing" else "pending"
            rows.append({**step.as_dict(), "status": state})
        return {
            "instruction": self.instruction,
            "intent": self.steps[self.current_step].action if self.current_step < len(self.steps) else "stop",
            "phase": self.phase, "reason": self.reason,
            "complete": self.complete, "failed": self.failed,
            "model": configured.get("model"), "provider": configured.get("provider"),
            "camera_png": self.latest_png, "raw_response": self.raw_response,
            "mission_steps": rows, "current_step": self.current_step,
            "inference_ms": self.inference_ms,
            "elapsed_s": float(self.env.data.time - self.started),
            "execution_sim_s": float(self.execution_s),
            "eye_grounding": self.eye_grounding,
            "replans": self.replans,
            "vision": "actual perspective robot-eye RGB for VLM planning",
            "limitation": "VLM plans from robot eyes; v4 low-level skills still use overhead RGB and analytic/PPO control.",
            "rows": [],
        }

    def close(self):
        self.closed = True
        if self.pending is not None:
            self.pending.cancel()
        self.skills.close()
        if self.eye is not None:
            self.eye.close()


class BrainRouter:
    """Keep the previous arena modes available alongside explicit generative mode."""

    def __init__(self, env):
        from robopup.vlm_runtime import get_vlm_runtime

        self.env = env
        self.fast = BrainSession(env)
        self.vlm = VLMMission(env, get_vlm_runtime())
        self.mode = "neural"

    @property
    def available(self):
        return self.fast.available

    @property
    def executor(self):
        return self.active.executor

    @property
    def active(self):
        return self.vlm if self.mode == "vlm" else self.fast

    def set_mode(self, mode):
        if mode == "vlm" and not self.vlm.runtime.status().get("available"):
            raise ValueError("Generative VLM unavailable; download/configure the model first")
        if self.mode == "vlm" and mode != "vlm":
            self.vlm.close()
            self.vlm = VLMMission(self.env, self.vlm.runtime)
        self.mode = mode

    def begin(self, instruction):
        self.active.begin(instruction)

    def cancel_vlm(self):
        self.vlm.close()
        self.vlm = VLMMission(self.env, self.vlm.runtime)

    def tick(self):
        return self.active.tick()

    def status(self):
        return self.active.status()

    def close(self):
        self.fast.close()
        self.vlm.close()
