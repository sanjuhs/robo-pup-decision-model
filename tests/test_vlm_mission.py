"""Mission contract tests with controlled runtime outputs, not LLM evidence.

The fetch/hop test runs the real MuJoCo environment, eye and overhead cameras,
trained v4 planner and PPO residuals. Only the high-level model response is a
clearly labeled fixture; passing this suite does not establish VLM competence.
"""

import asyncio
import base64
from collections import deque
import hashlib
import io
from pathlib import Path
import threading

import mujoco
import numpy as np
from PIL import Image
import pytest
from stable_baselines3 import PPO

from robopup.pet import PetEnv
from robopup.vlm_mission import BrainRouter, VLMMission
from robopup.vlm_runtime import VLMGeneration

ROOT = Path(__file__).resolve().parents[1]

# Recorded-plan-shaped protocol fixtures, deliberately not attributed to an LLM.
LOOK_STOP_FIXTURE = '{"summary":"Observe, then stop.","steps":[{"action":"look","object":"none"},{"action":"stop","object":"none"}]}'
STOP_FIXTURE = '{"summary":"Stop.","steps":[{"action":"stop","object":"none"}]}'
FETCH_HOP_FIXTURE = '{"summary":"Fetch the ball home, then hop.","steps":[{"action":"fetch","object":"ball"},{"action":"jump","object":"none"}]}'
HOP_FIXTURE = '{"summary":"The fetch is complete; now hop.","steps":[{"action":"jump","object":"none"}]}'


class ControlledRuntime:
    """Deterministic test double; never used as an application model fallback."""

    def __init__(self, outputs, *, gate=None, available=True):
        self.outputs = deque(outputs)
        self.gate = gate
        self.available = available
        self.calls = []
        self.entered = threading.Event()

    def status(self):
        return {"available": self.available, "model": "controlled-contract-fixture",
                "provider": "test-double", "reason": "Controlled runtime unavailable"}

    def generate(self, image, instruction, *, energy, held_object):
        self.calls.append({"image": image.copy(), "instruction": instruction,
                           "energy": energy, "held_object": held_object})
        self.entered.set()
        if self.gate is not None and not self.gate.wait(timeout=5):
            raise RuntimeError("Controlled test gate timed out")
        output = self.outputs.popleft()
        if isinstance(output, Exception):
            raise output
        return VLMGeneration(raw_text=output, model="controlled-contract-fixture",
                             provider="test-double", timing_ms={"total": 1.25},
                             metadata={"test_fixture": True})


def settled_env():
    env = PetEnv(episode_seconds=400)
    env.reset(seed=7)
    env.command[:] = 0
    for _ in range(25):
        env.step(np.zeros(12))
    return env


async def wait_for_generation(mission):
    assert mission.pending is not None
    # Retrieve possible exceptions here, then tick exercises the runtime's own
    # sanitized failure handling through future.result().
    await asyncio.wait_for(asyncio.gather(mission.pending, return_exceptions=True), timeout=5)


def test_two_step_look_stop_replans_from_actual_eye_and_keeps_mission_incomplete_between_steps():
    async def scenario():
        env = settled_env()
        runtime = ControlledRuntime([LOOK_STOP_FIXTURE, STOP_FIXTURE])
        mission = VLMMission(env, runtime)
        try:
            mission.begin("look around and then stop")
            assert mission.phase == "thinking"
            await wait_for_generation(mission)
            mission.tick()
            assert mission.phase == "executing" and mission.current_step == 0
            assert not mission.complete
            first_step_start = mission.step_started
            while env.data.time - first_step_start < 0.45:
                env.command[:], learned = mission.tick()
                assert not learned
                env.step(np.zeros(12))
            assert mission.current_step == 0
            while mission.phase != "thinking" and env.data.time - first_step_start < 2:
                env.command[:], _ = mission.tick()
                env.step(np.zeros(12))
            assert mission.phase == "thinking", mission.reason
            assert mission.current_step == 1 and mission.replans == 1
            assert not mission.complete
            assert mission.status()["mission_steps"][0]["status"] == "complete"
            await wait_for_generation(mission)
            assert len(runtime.calls) == 2
            call = runtime.calls[0]
            assert call["image"].shape == (384, 384, 3)
            assert call["image"].dtype == np.uint8 and call["image"].std() > 10
            assert set(call) == {"image", "instruction", "energy", "held_object"}
            assert type(call["held_object"]) is bool
            assert "Verified completed steps" in runtime.calls[1]["instruction"]
            assert '"action": "look"' in runtime.calls[1]["instruction"]
            png = Image.open(io.BytesIO(base64.b64decode(mission.latest_png)))
            np.testing.assert_array_equal(np.asarray(png), runtime.calls[1]["image"])
            for _ in range(100):
                env.command[:], _ = mission.tick()
                env.step(np.zeros(12))
                if mission.complete or mission.failed:
                    break
            assert mission.complete and not mission.failed, mission.reason
            assert mission.current_step == 2 and len(mission.history) == 2
            assert all(step["status"] == "complete" for step in mission.status()["mission_steps"])
            for history, observed in zip(mission.history, runtime.calls):
                assert history["eye_rgb_sha256"] == hashlib.sha256(observed["image"].tobytes()).hexdigest()
                assert history["metadata"] == {"test_fixture": True}
            np.testing.assert_array_equal(mission.tick()[0], np.zeros(2))
        finally:
            mission.close()
            env.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("output", ["not valid mission JSON", ValueError("private-provider-diagnostic")])
def test_runtime_and_schema_failures_stop_without_model_fallback_or_exception_leaks(output):
    async def scenario():
        env = settled_env()
        runtime = ControlledRuntime([output])
        mission = VLMMission(env, runtime)
        try:
            mission.begin("fetch the ball")
            await wait_for_generation(mission)
            command, learned = mission.tick()
            np.testing.assert_array_equal(command, np.zeros(2))
            assert not learned and mission.failed and not mission.complete
            assert mission.phase == "error" and mission.steps == []
            assert mission.reason == "VLM request or structured plan validation failed; no action executed"
            assert "private-provider-diagnostic" not in mission.reason
            assert mission.executor.instruction == "stop"
            assert not env.magnet and not env.held
            assert len(runtime.calls) == 1
        finally:
            mission.close()
            env.close()

    asyncio.run(scenario())


def test_pending_generation_is_busy_and_close_cancels_its_future_without_starting_actions():
    async def scenario():
        env = settled_env()
        gate = threading.Event()
        runtime = ControlledRuntime([FETCH_HOP_FIXTURE], gate=gate)
        mission = VLMMission(env, runtime)
        try:
            mission.begin("fetch then hop")
            assert await asyncio.to_thread(runtime.entered.wait, 2)
            pending = mission.pending
            with pytest.raises(ValueError, match="still running"):
                mission.begin("run forward")
            np.testing.assert_array_equal(mission.tick()[0], np.zeros(2))
            assert mission.executor.instruction == "stop" and not env.magnet
            mission.close()
            assert mission.closed and pending.cancelled()
            gate.set()
            np.testing.assert_array_equal(mission.tick()[0], np.zeros(2))
            assert mission.steps == [] and not mission.complete
        finally:
            gate.set()
            mission.close()
            env.close()

    asyncio.run(scenario())


def test_unavailable_runtime_does_not_construct_camera_or_claim_a_plan():
    env = settled_env()
    runtime = ControlledRuntime([], available=False)
    mission = VLMMission(env, runtime)
    try:
        with pytest.raises(ValueError, match="Controlled runtime unavailable"):
            mission.begin("fetch the ball")
        assert mission.pending is None and mission.eye is None and mission.plan is None
        assert runtime.calls == []
    finally:
        mission.close()
        env.close()


def test_mission_budget_timeout_releases_gripper_and_never_claims_success():
    async def scenario():
        env = settled_env()
        mission = VLMMission(env, ControlledRuntime([LOOK_STOP_FIXTURE]))
        try:
            mission.begin("look then stop")
            await wait_for_generation(mission)
            mission.tick()
            env.grip(True)
            # Instrument the clock to exercise the safety guard without a
            # six-minute simulation; this is not a behavior success test.
            env.data.time = mission.started + 360.01
            command, learned = mission.tick()
            np.testing.assert_array_equal(command, np.zeros(2))
            assert not learned and mission.failed and not mission.complete
            assert mission.reason == "Mission timed out" and not env.magnet
        finally:
            mission.close()
            env.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("exhausted_replans", [False, True])
def test_replan_failure_or_limit_never_executes_the_stale_remaining_plan(exhausted_replans):
    async def scenario():
        env = settled_env()
        runtime = ControlledRuntime([LOOK_STOP_FIXTURE, "invalid second plan"])
        mission = VLMMission(env, runtime)
        try:
            mission.begin("look then stop")
            await wait_for_generation(mission)
            mission.tick()
            if exhausted_replans:
                # Instrument the budget counter rather than asking for eight
                # redundant model calls; the first direct skill is still real.
                mission.replans = 7
            start = env.data.time
            while not mission.failed and env.data.time - start < 2:
                if mission.pending is not None:
                    await wait_for_generation(mission)
                env.command[:], _ = mission.tick()
                env.step(np.zeros(12))
            assert mission.failed and not mission.complete
            assert mission.current_step == 1
            assert mission.executor.instruction == "stop"
            assert len(mission.history) == 1
            if exhausted_replans:
                assert mission.reason == "VLM replan limit reached"
                assert len(runtime.calls) == 1
            else:
                assert "structured plan validation failed" in mission.reason
                assert len(runtime.calls) == 2
            np.testing.assert_array_equal(mission.tick()[0], np.zeros(2))
        finally:
            mission.close()
            env.close()

    asyncio.run(scenario())


def test_fallen_robot_never_executes_a_generated_step():
    async def scenario():
        env = settled_env()
        mission = VLMMission(env, ControlledRuntime([FETCH_HOP_FIXTURE]))
        try:
            mission.begin("fetch then hop")
            await wait_for_generation(mission)
            env.data.qpos[2] = 0.1
            mujoco.mj_forward(env.model, env.data)
            command, learned = mission.tick()
            np.testing.assert_array_equal(command, np.zeros(2))
            assert not learned and mission.failed and not mission.complete
            assert mission.reason == "Robot fell; reset to continue"
            assert mission.executor.instruction == "stop"
        finally:
            mission.close()
            env.close()

    asyncio.run(scenario())


def test_router_switch_cancels_vlm_and_does_not_replace_the_fast_mode(monkeypatch):
    async def scenario():
        env = settled_env()
        gate = threading.Event()
        runtime = ControlledRuntime([FETCH_HOP_FIXTURE], gate=gate)
        monkeypatch.setattr("robopup.vlm_runtime.get_vlm_runtime", lambda: runtime)
        router = BrainRouter(env)
        try:
            fast = router.fast
            router.set_mode("vlm")
            router.begin("fetch then hop")
            assert await asyncio.to_thread(runtime.entered.wait, 2)
            previous = router.vlm
            pending = previous.pending
            router.set_mode("neural")
            assert previous.closed and pending.cancelled()
            assert router.fast is fast and router.active is fast
            assert router.vlm is not previous
            assert router.vlm.pending is None and router.vlm.plan is None
            gate.set()
        finally:
            gate.set()
            router.close()
            env.close()

    asyncio.run(scenario())


def test_explicit_router_cancel_discards_pending_mission_while_retaining_vlm_mode(monkeypatch):
    async def scenario():
        env = settled_env()
        gate = threading.Event()
        runtime = ControlledRuntime([FETCH_HOP_FIXTURE], gate=gate)
        monkeypatch.setattr("robopup.vlm_runtime.get_vlm_runtime", lambda: runtime)
        router = BrainRouter(env)
        try:
            router.set_mode("vlm")
            router.begin("fetch then hop")
            assert await asyncio.to_thread(runtime.entered.wait, 2)
            previous = router.vlm
            pending = previous.pending
            router.cancel_vlm()
            assert previous.closed and pending.cancelled()
            assert router.mode == "vlm" and router.active is router.vlm
            assert router.vlm is not previous and router.vlm.plan is None
            gate.set()
            command, learned = router.tick()
            np.testing.assert_array_equal(command, np.zeros(2))
            assert not learned and router.status()["phase"] == "idle"
        finally:
            gate.set()
            router.close()
            env.close()

    asyncio.run(scenario())


def test_contract_fixture_fetch_then_hop_runs_real_physics_and_completes_only_after_both_steps():
    """Real skill integration under fixture planning; this is not VLM evidence."""
    async def scenario():
        env = settled_env()
        runtime = ControlledRuntime([FETCH_HOP_FIXTURE, HOP_FIXTURE])
        mission = VLMMission(env, runtime)
        motor = PPO.load(ROOT / "demo/curious-v2/policy.zip", device="cpu")
        fetch_evidence = None
        try:
            mission.begin("fetch the red ball and then hop")
            while env.data.time < 130 and not mission.complete and not mission.failed:
                if mission.pending is not None:
                    await wait_for_generation(mission)
                command, learned = mission.tick()
                if mission.current_step == 1 and fetch_evidence is None:
                    # The low-level fetch finished, but the multi-step mission
                    # must remain incomplete until the separately verified hop.
                    assert mission.executor.complete and not mission.complete
                    assert mission.phase == "thinking"
                    fetch_evidence = {
                        "held": mission.executor.ever_held,
                        "lift": mission.executor.maximum_ball_height,
                        "ball": env.data.xpos[env.ball].copy(),
                        "eye": runtime.calls[0]["image"].copy(),
                    }
                env.command[:] = command
                action = motor.predict(env._obs(), deterministic=True)[0] if learned else np.zeros(12)
                env.step(action)
            assert mission.complete and not mission.failed, mission.reason
            assert mission.current_step == 2 and mission.replans == 1
            assert fetch_evidence is not None
            assert fetch_evidence["held"] and fetch_evidence["lift"] > 0.2
            assert np.linalg.norm(fetch_evidence["ball"][:2]) < 0.6
            assert fetch_evidence["ball"][2] < 0.19
            assert env.grasp_count == 1 and not env.held
            assert env.jump_flight_seconds > 0.04
            assert env.jump_time is None and len(env.foot_contacts) >= 2 and env.upright > 0.9
            assert len(runtime.calls) == 2 and len(mission.history) == 2
            assert '"action": "fetch"' in runtime.calls[1]["instruction"]
            assert not np.array_equal(runtime.calls[0]["image"], runtime.calls[1]["image"])
            assert all(row["status"] == "complete" for row in mission.status()["mission_steps"])
        finally:
            mission.close()
            env.close()

    asyncio.run(scenario())
