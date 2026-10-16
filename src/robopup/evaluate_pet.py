"""Actual physics goal trials; optional oracle is an explicitly separate baseline."""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO

from robopup.executor import GoalExecutor, heading
from robopup.pet import PetEnv
from robopup.planner_data import labels
from robopup.scene_planner import PLAN_SKILLS, infer, load_planner
from robopup.vision import PlannerCamera

COMMANDS = {
    "approach": "go to the ball",
    "behind": "go behind the ball",
    "fetch": "fetch the ball",
    "push": "push the crate",
    "follow": "follow me",
    "walk": "walk forward",
    "run": "run forward",
    "jump": "jump",
    "rest": "rest",
}


def oracle_plan(env, intent):
    """Mechanics diagnostic only. Never imported by the live arena."""
    skills, objects, offsets = labels(intent, env.energy, env.held)
    positions = [
        env.data.xpos[env.ball, :2],
        env.data.xpos[env.crate, :2],
        np.zeros(2),
        env.data.mocap_pos[0, :2],
        env.data.qpos[:2],
    ]
    rows = []
    for skill, obj, offset in zip(skills, objects, offsets):
        delta = positions[obj] - env.data.qpos[:2]
        rows.append(
            {
                "skill": PLAN_SKILLS[skill],
                "object": ("ball", "crate", "home", "beacon", "none")[obj],
                "delta": (delta + offset).tolist(),
                "object_delta": delta.tolist(),
                "confidence": 1.0,
                "visible": True,
                "duration_s": 1.0,
            }
        )
    return {"intent": intent, "confidence": 1.0, "rows": rows}


def evaluate(model_dir, out, intents, seeds, seconds=150, oracle=False, motor=True):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError("Keep previous trials; choose a new output filename")
    torch.set_num_threads(1)
    model = None if oracle else load_planner(model_dir)[0]
    root = Path(__file__).resolve().parents[2]
    policy = PPO.load(root / "demo/curious-v2/policy.zip", device="cpu") if motor else None
    results = []
    wall = time.monotonic()
    for intent in intents:
        for seed in seeds:
            env = PetEnv(episode_seconds=seconds + 5)
            env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            yaw = rng.uniform(-0.12, 0.12)
            env.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            for name in ("ball_free", "crate_free"):
                joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                env.data.qpos[env.model.jnt_qposadr[joint] : env.model.jnt_qposadr[joint] + 2] += (
                    rng.uniform(-0.15, 0.15, 2)
                )
            mujoco.mj_forward(env.model, env.data)
            env.command[:] = 0
            for _ in range(25):
                env.step(np.zeros(12))
            ex = GoalExecutor(env)
            ex.begin(COMMANDS[intent])
            camera = None if oracle else PlannerCamera(env)
            trace = []
            start = env.data.time
            try:
                while env.data.time - start < seconds:
                    if intent == "follow":
                        from robopup.decisions import follow_target

                        env.data.mocap_pos[0, :2] = follow_target(env.data.time)
                    if env.data.time - ex.last_plan >= 0.5:
                        plan = (
                            oracle_plan(env, intent)
                            if oracle
                            else infer(
                                model,
                                camera.capture(),
                                ex.instruction,
                                env._obs(),
                                env.energy,
                                float(env.held),
                                heading(env),
                            )
                        )
                        ex.update_plan(plan)
                        trace.append(
                            {
                                "time": round(env.data.time - start, 2),
                                "intent": plan["intent"],
                                "row": plan["rows"][0],
                                "position": env.data.qpos[:3].tolist(),
                                "phase": ex.phase,
                                "held": env.held,
                            }
                        )
                    env.command[:] = ex.command()
                    learned = (
                        policy is not None
                        and env.motion == "trot"
                        and abs(env.command[1]) < 0.25
                        and env.jump_time is None
                        and np.linalg.norm(env.command) > 0.03
                    )
                    action = (
                        policy.predict(env._obs(), deterministic=True)[0]
                        if learned
                        else np.zeros(12)
                    )
                    _, _, fallen, _, _ = env.step(action)
                    if ex.complete or ex.failed or fallen:
                        break
                    if intent == "follow" and env.data.time - start >= 35:
                        break
                ball = env.data.xpos[env.ball].copy()
                follow_error = float(np.linalg.norm(env.data.qpos[:2] - env.data.mocap_pos[0, :2]))
                success = (
                    bool(ex.complete)
                    if intent != "follow"
                    else follow_error < 0.75 and env.upright > 0.9
                )
                result = {
                    "intent": intent,
                    "seed": seed,
                    "success": success,
                    "seconds": env.data.time - start,
                    "position": env.data.qpos[:3].tolist(),
                    "upright": env.upright,
                    "ball": ball.tolist(),
                    "ball_touched": env.ball_touched,
                    "held": env.held,
                    "ever_held": ex.ever_held,
                    "maximum_ball_height": ex.maximum_ball_height,
                    "grasp_count": env.grasp_count,
                    "jump_max_height": env.jump_max_height,
                    "jump_flight_seconds": env.jump_flight_seconds,
                    "gait_air_seconds": env.gait_air_seconds.copy(),
                    "follow_error_m": follow_error,
                    "reason": ex.reason,
                    "trace": trace,
                }
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k != "trace"}), flush=True)
                out.write_text(
                    json.dumps(
                        {
                            "oracle": oracle,
                            "motor": "PPO hybrid" if motor else "analytic",
                            "wall_seconds": time.monotonic() - wall,
                            "trials": results,
                        },
                        indent=2,
                    )
                    + "\n"
                )
            finally:
                if camera:
                    camera.close()
                env.close()
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("demo/arena-planner"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--analytic", action="store_true")
    p.add_argument("--seconds", type=float, default=150)
    p.add_argument(
        "--intents",
        nargs="+",
        choices=list(COMMANDS),
        default=["approach", "behind", "fetch", "push", "follow", "walk", "run", "jump"],
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[123, 124, 125])
    a = p.parse_args()
    evaluate(a.model, a.out, a.intents, a.seeds, a.seconds, a.oracle, not a.analytic)
