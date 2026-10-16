"""Reproducible rendered training scenes and oracle labels; never live perception."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from robopup.pet import PetEnv
from robopup.scene_planner import PLAN_SKILLS
from robopup.vision import PlannerCamera

PHRASES = {
    "stop": ["stop", "stay", "wait", "halt", "stand still", "stay here"],
    "approach": [
        "go to the ball",
        "boop the ball",
        "touch the ball",
        "eat ball",
        "sniff the red ball",
        "approach the ball",
    ],
    "behind": [
        "go behind the ball",
        "walk behind ball",
        "get behind the red ball",
        "go to the far side of the ball",
    ],
    "fetch": [
        "fetch the ball",
        "bring the ball home",
        "get the ball and return",
        "collect the ball",
        "retrieve the ball",
        "bring me the ball",
    ],
    "follow": [
        "follow me",
        "follow the beacon",
        "follow along",
        "track the purple beacon",
        "chase the beacon",
    ],
    "push": [
        "push the crate",
        "push the box",
        "nudge the crate",
        "move the orange crate",
        "push it to the ring",
    ],
    "walk": ["walk", "walk forward", "take a walk", "go slowly", "slow walk"],
    "run": ["run", "run forward", "sprint", "go fast", "zoomies"],
    "jump": ["jump", "hop", "do a jump", "leap", "jump up"],
    "rest": ["rest", "take a break", "recharge", "relax", "have a rest"],
    "drop": ["drop the ball", "release the ball", "let go of the ball", "put the ball down"],
    "unsupported": [
        "what is the weather",
        "tell me a joke",
        "hello",
        "fly to the moon",
        "what color is the ball",
        "delete files",
        "open the door",
    ],
}
HELDOUT = {
    "stop": "please stay still here",
    "approach": "go and touch the red ball",
    "behind": "please go to the far side behind ball",
    "fetch": "please fetch and bring the ball home",
    "follow": "please follow the purple beacon",
    "push": "please nudge the orange box",
    "walk": "miso walk slowly forward",
    "run": "miso please run fast",
    "jump": "please do a hop",
    "rest": "please take a rest break",
    "drop": "please release and drop the ball",
    "unsupported": "what is your favorite color",
}


def labels(intent, energy, held):
    skills = np.zeros(6, dtype=np.int64)
    objects = np.full(6, 4, dtype=np.int64)
    offsets = np.zeros((6, 2), dtype=np.float32)
    if energy < 0.15 and intent not in ("stop", "drop", "unsupported"):
        skills[:] = PLAN_SKILLS.index("rest")
    elif intent == "fetch":
        if not held:
            skills[:3] = [PLAN_SKILLS.index("retrieve"), PLAN_SKILLS.index("return"), 0]
            objects[:3] = [0, 2, 4]
        else:
            skills[:2] = [PLAN_SKILLS.index("return"), PLAN_SKILLS.index("drop")]
            objects[:2] = [2, 4]
    elif intent in ("approach", "behind", "follow", "push"):
        skills[:] = PLAN_SKILLS.index(intent)
        objects[:] = {"approach": 0, "behind": 0, "follow": 3, "push": 1}[intent]
        if intent == "behind":
            offsets[:, 0] = 0.65
    elif intent in PLAN_SKILLS:
        skills[:] = PLAN_SKILLS.index(intent)
    return skills, objects, offsets


def generate(out: Path, scenes=1800, seed=61, rollout_steps=0):
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    env = PetEnv()
    env.reset(seed=seed)
    camera = PlannerCamera(env)
    images, masks, observations = [], [], []
    ball_joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    crate_joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "crate_free")
    start = time.monotonic()
    try:
        for index in range(scenes):
            env.reset(seed=seed)
            xy = rng.uniform(-1.3, 1.3, 2)
            yaw = rng.uniform(-np.pi, np.pi)
            env.data.qpos[:2] = xy
            env.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
            for joint, height in ((ball_joint, 0.125), (crate_joint, 0.14)):
                adr = env.model.jnt_qposadr[joint]
                relative = rng.uniform(-2.3, 2.3, 2)
                if np.linalg.norm(relative) < 0.6:
                    relative[0] += 0.8
                env.data.qpos[adr : adr + 3] = [*(xy + relative), height]
            env.data.mocap_pos[0, :2] = xy + rng.uniform(-2.3, 2.3, 2)
            for name, base in (
                ("ball_geom", [0.96, 0.44, 0.34]),
                ("crate_geom", [1.0, 0.6, 0.2]),
                ("home_pad", [0.12, 0.38, 1.0]),
                ("beacon_geom", [0.75, 0.15, 0.9]),
            ):
                gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                env.model.geom_rgba[gid, :3] = np.clip(
                    np.array(base) + rng.uniform(-0.045, 0.045, 3), 0, 1
                )
            env.model.light_diffuse[:] = rng.uniform(0.65, 1.0)
            env.data.qvel[:6] = rng.normal(0, 0.05, 6)
            env.command[:] = rng.uniform([0, -0.2], [0.45, 0.2])
            mujoco.mj_forward(env.model, env.data)
            for _ in range(rollout_steps):
                env.step(np.zeros(12))
            image, mask = camera.capture(labels=True)
            images.append(image)
            masks.append(mask)
            observations.append(env._obs())
            if (index + 1) % 300 == 0:
                print(f"Rendered {index + 1}/{scenes} labelled scenes", flush=True)
    finally:
        camera.close()
        env.close()
    path = out / "scenes.npz"
    np.savez_compressed(
        path, images=np.array(images), masks=np.array(masks), observations=np.array(observations)
    )
    report = {
        "seed": seed,
        "scenes": scenes,
        "rollout_steps_per_scene": rollout_steps,
        "seconds": time.monotonic() - start,
        "source": "actual orthographic MuJoCo RGB and segmentation; random scene poses",
        "split": "first 80% train; remaining scenes validation; separate held-out instruction templates",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "privileged_labels": "Segmentation IDs are training labels only; no object coordinates enter live inference",
    }
    (out / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", type=int, default=1800)
    p.add_argument("--seed", type=int, default=61)
    p.add_argument("--rollout-steps", type=int, default=0)
    a = p.parse_args()
    print(json.dumps(generate(a.out, a.scenes, a.seed, a.rollout_steps), indent=2))
