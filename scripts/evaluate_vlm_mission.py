#!/usr/bin/env python3
"""Actual generative VLM -> verified sequential native-physics mission trial.

No plans are supplied by this evaluator. It records the real eye image before
each actual model call, executes VLMMission, and independently checks physical
outcomes. Native MuJoCo remains CPU physics, even when VLM inference uses CUDA.
Simulation time is explicitly frozen while the VLM thinks; this is a headless
correctness trial, not a measurement of real-time browser responsiveness.
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image

from robopup.vlm_protocol import MissionValidationError, build_planning_prompt, parse_mission
from robopup.vlm_runtime import VLMRuntime, VLMRuntimeError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INSTRUCTION = "Fetch the red ball, then hop, then stop."


def save_json(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


class RecordingRuntime:
    """Transparent recorder: every response comes from the delegated real VLM."""

    def __init__(self, runtime, out):
        self.runtime, self.out = runtime, out
        self._records, self._lock = [], threading.Lock()

    def status(self):
        return self.runtime.status()

    def records(self):
        with self._lock:
            return list(self._records)

    def generate(self, image, instruction, energy, held_object):
        with self._lock:
            index = len(self._records)
            self._records.append({"index": index, "phase": "requested"})
        image = image if isinstance(image, Image.Image) else Image.fromarray(image)
        image = image.convert("RGB")
        image_path = self.out / f"eye-{index:03}.png"
        image.save(image_path)
        record = {
            "index": index, "phase": "requested", "instruction": instruction,
            "energy": energy, "held": held_object, "image": image_path.name,
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "eye_rgb_sha256": hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),
            "image_saved_before_model_call": True,
            "actual_generated_response_received": False, "valid_mission": False,
        }
        output = self.out / f"inference-{index:03}.json"
        save_json(output, record)
        try:
            result = self.runtime.generate(image, instruction, energy, held_object)
            record.update(phase="response", actual_generated_response_received=True,
                          generation=result.as_dict())
            try:
                record.update(mission=parse_mission(result.raw_text).as_dict(), valid_mission=True)
            except MissionValidationError as error:
                record["validation_error"] = str(error)
            return result
        except Exception as error:
            record["phase"] = "error"
            record["error_type"] = type(error).__name__
            record["error"] = (str(error) if isinstance(error, VLMRuntimeError)
                               else "Model request failed; exception body withheld")
            raise
        finally:
            save_json(output, record)
            with self._lock:
                self._records[index] = record


async def evaluate(args):
    # Import physics and Torch only for an explicitly requested evaluation.
    import torch
    from stable_baselines3 import PPO
    from robopup.brain import trained_model
    from robopup.pet import PetEnv
    from robopup.vlm_mission import VLMMission

    args.out.mkdir(parents=True, exist_ok=False)
    build_planning_prompt(args.instruction, energy=.9, held=False)
    started = time.monotonic()
    record = {
        "schema": "robopup-real-vlm-physical-mission-v1", "seed": args.seed,
        "instruction": args.instruction, "complete": False, "success": False,
        "physics": "native MuJoCo CPU", "physics_hz": 50,
        "motor": "bundled curious-v2 CPU PPO residuals plus analytic skill controllers",
        "high_level_vision": "actual perspective robot-head RGB attached to every real VLM call",
        "low_level_vision": "v4 skills retain their evaluated overhead RGB camera",
        "simulation_paused_during_vlm_inference": True,
        "real_time_responsiveness_tested": False,
        "maximum_sim_seconds": args.sim_seconds, "maximum_wall_seconds": args.wall_seconds,
        "steps_verified": [], "thinking_intervals": [], "trace": [],
        "metrics": {"ever_ball_touched": False, "ever_held": False,
                    "maximum_ball_height": 0.0, "maximum_held_ball_height": 0.0,
                    "release_events": 0, "ppo_residual_steps": 0,
                    "external_force_ever_applied": False},
    }
    result_path = args.out / "result.json"
    save_json(result_path, record)
    runtime = RecordingRuntime(VLMRuntime(), args.out)
    env = mission = None
    thinking = None
    try:
        torch.set_num_threads(1)
        policy = PPO.load(ROOT / "demo/curious-v2/policy.zip", device="cpu")
        trained_model()  # Warm v4 before the VLM configures its own CPU thread count.
        env = PetEnv(episode_seconds=args.sim_seconds + 5)
        env.reset(seed=args.seed)
        env.command[:] = 0
        for _ in range(25):
            env.step(np.zeros(12))
        env.energy = .9
        mission = VLMMission(env, runtime)
        mission.begin(args.instruction)
        sim_start = float(env.data.time)
        record["settle_steps"] = 25
        record["sim_start"] = sim_start
        record["initial_runtime_status"] = runtime.status()
        last_held = bool(env.held)
        physics_steps = 0
        verified_count = 0
        last_phase = None
        last_save_wall = 0.0
        limit_reason = None
        while not (mission.complete or mission.failed):
            if time.monotonic() - started >= args.wall_seconds:
                limit_reason = "Evaluator wall-time limit reached; no completion claimed"
                break
            if env.data.time - sim_start >= args.sim_seconds:
                limit_reason = "Evaluator simulation-time limit reached; no completion claimed"
                break
            command, learned_motor = mission.tick()
            if mission.current_step > verified_count:
                for index in range(verified_count, mission.current_step):
                    step = mission.steps[index]
                    record["steps_verified"].append({
                        **step.as_dict(), "sim_seconds": float(env.data.time - sim_start),
                        "executor": mission.executor.status(),
                        "ball": env.data.xpos[env.ball].tolist(), "held": bool(env.held),
                        "grasp_count": env.grasp_count, "upright": env.upright,
                        "jump_active": env.jump_time is not None,
                        "jump_flight_seconds": env.jump_flight_seconds,
                        "foot_contacts": len(env.foot_contacts),
                    })
                verified_count = mission.current_step
            if mission.phase == "thinking":
                if thinking is None:
                    thinking = {"wall_start": time.monotonic() - started,
                                "sim_start": float(env.data.time)}
                await asyncio.sleep(.01)
            else:
                if thinking is not None:
                    thinking.update(wall_seconds=time.monotonic() - started - thinking["wall_start"],
                                    sim_end=float(env.data.time))
                    record["thinking_intervals"].append(thinking)
                    thinking = None
                if not (mission.complete or mission.failed):
                    env.command[:] = command
                    use_ppo = bool(learned_motor and np.linalg.norm(command) > .03)
                    action = (policy.predict(env._obs(), deterministic=True)[0]
                              if use_ppo else np.zeros(12))
                    _, _, fallen, truncated, _ = env.step(action)
                    physics_steps += 1
                    metrics = record["metrics"]
                    held = bool(env.held)
                    ball_height = float(env.data.xpos[env.ball, 2])
                    metrics["ever_ball_touched"] |= bool(env.ball_touched)
                    metrics["ever_held"] |= held
                    metrics["maximum_ball_height"] = max(metrics["maximum_ball_height"], ball_height)
                    if held:
                        metrics["maximum_held_ball_height"] = max(
                            metrics["maximum_held_ball_height"], ball_height,
                        )
                    metrics["release_events"] += int(last_held and not held)
                    metrics["ppo_residual_steps"] += int(use_ppo)
                    metrics["external_force_ever_applied"] |= bool(
                        np.any(env.data.xfrc_applied) or np.any(env.data.qfrc_applied)
                    )
                    last_held = held
                    if fallen or truncated:
                        limit_reason = "Physics termination or episode limit; no completion claimed"
                        break
                    await asyncio.sleep(0)
            if (mission.phase != last_phase
                    or (mission.phase != "thinking" and physics_steps % 50 == 0)
                    or time.monotonic() - last_save_wall >= 2):
                status = mission.status()
                status.pop("camera_png", None)
                record["mission"] = status
                record["wall_seconds"] = time.monotonic() - started
                record["sim_seconds"] = float(env.data.time - sim_start)
                if mission.phase != last_phase:
                    record["trace"].append({"wall_seconds": record["wall_seconds"],
                                            "sim_seconds": record["sim_seconds"],
                                            "phase": mission.phase,
                                            "current_step": mission.current_step})
                    print("ROBOPUP_STAGE:" + json.dumps({"mission_phase": mission.phase,
                          "current_step": mission.current_step,
                          "sim_seconds": record["sim_seconds"]}), flush=True)
                    last_phase = mission.phase
                save_json(result_path, record)
                last_save_wall = time.monotonic()
        if thinking is not None:
            thinking.update(wall_seconds=time.monotonic() - started - thinking["wall_start"],
                            sim_end=float(env.data.time))
            record["thinking_intervals"].append(thinking)
        record.update(complete=bool(mission.complete), failed=bool(mission.failed),
                      reason=limit_reason or mission.reason, history=mission.history,
                      wall_seconds=time.monotonic() - started,
                      sim_seconds=float(env.data.time - sim_start), physics_steps=physics_steps,
                      model_calls=runtime.records(), final_runtime_status=runtime.status())
        record["metrics"].update(grasp_count=env.grasp_count, held_final=bool(env.held),
                                 ball_final=env.data.xpos[env.ball].tolist(),
                                 jump_max_height=env.jump_max_height,
                                 jump_flight_seconds=env.jump_flight_seconds,
                                 jump_active=env.jump_time is not None,
                                 foot_contacts_final=len(env.foot_contacts), upright_final=env.upright)
        metrics = record["metrics"]
        completed_actions = [step["action"] for step in record["steps_verified"]]
        fetches = [step for step in record["steps_verified"] if step["action"] == "fetch"]
        jumps = [step for step in record["steps_verified"] if step["action"] == "jump"]
        checks = {
            "mission_complete": bool(mission.complete and not mission.failed and not limit_reason),
            "requested_actions_verified_in_order": completed_actions == ["fetch", "jump", "stop"],
            "every_response_from_actual_model": bool(record["model_calls"]) and all(
                call.get("actual_generated_response_received") and call.get("valid_mission")
                for call in record["model_calls"]),
            "recorded_eyes_match_mission_observations": (
                len(record["model_calls"]) == len(mission.history)
                and all(call.get("eye_rgb_sha256") == entry.get("eye_rgb_sha256")
                        for call, entry in zip(record["model_calls"], mission.history))
            ),
            "replanning_after_each_step": len(record["model_calls"]) >= 3,
            "contact_and_grasp": bool(metrics["ever_ball_touched"] and metrics["ever_held"]
                                      and env.grasp_count >= 1),
            "lift": metrics["maximum_held_ball_height"] > .2,
            "return_and_release": bool(fetches and metrics["release_events"] >= 1
                and not env.held and np.linalg.norm(fetches[0]["ball"][:2]) < .6
                and fetches[0]["ball"][2] < .19),
            "hop_flight_and_landing": bool(jumps and jumps[0]["jump_flight_seconds"] > .04
                and jumps[0]["foot_contacts"] >= 2 and jumps[0]["upright"] > .9
                and not jumps[0]["jump_active"]),
            "ppo_residuals_used": metrics["ppo_residual_steps"] > 0,
            "no_external_forces": not metrics["external_force_ever_applied"],
            "physics_frozen_while_thinking": bool(record["thinking_intervals"]) and all(
                interval["sim_start"] == interval["sim_end"]
                for interval in record["thinking_intervals"]),
        }
        record.update(checks=checks, success=all(checks.values()))
    except Exception as error:
        record.update(error_type=type(error).__name__, success=False,
                      error=(str(error) if isinstance(error, (VLMRuntimeError, ValueError))
                             else "Evaluation failed; exception body withheld"),
                      wall_seconds=time.monotonic() - started, model_calls=runtime.records())
    finally:
        save_json(result_path, record)
        if mission is not None:
            mission.close()
        if env is not None:
            env.close()
    print(json.dumps({"output": str(result_path), "success": record["success"],
                      "complete": record["complete"], "wall_seconds": record["wall_seconds"]}),
          flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--sim-seconds", type=float, default=360)
    parser.add_argument("--wall-seconds", type=float, default=600)
    args = parser.parse_args()
    if not 1 <= args.sim_seconds <= 360 or not 1 <= args.wall_seconds <= 600:
        parser.error("Simulation limit must be 1..360s and wall limit 1..600s")
    if args.instruction != DEFAULT_INSTRUCTION:
        parser.error("This evidence protocol checks the fixed fetch, hop, stop mission")
    if not asyncio.run(evaluate(args))["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
