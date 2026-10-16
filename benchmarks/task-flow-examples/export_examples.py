"""Export actual v4 predictions from one frozen RGB frame, without task execution.

Held/low-energy examples explicitly intervene on model inputs. They are not
different physical scenes, physical grasps, or held-out evaluation results.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
from PIL import Image
import torch

from robopup.executor import heading
from robopup.pet import PetEnv
from robopup.scene_planner import infer, load_planner
from robopup.vision import PlannerCamera


REPO = Path(__file__).resolve().parents[2]
SEED = 77
SETTLE_STEPS = 100


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="New output directory; existing directories are refused",
    )
    args = parser.parse_args()
    output = args.out.resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error("Output directory already exists; choose a new --out path")
    torch.set_num_threads(1)
    directory = REPO / "demo/arena-planner"
    model, checkpoint = load_planner(directory)
    env = PetEnv()
    camera = None
    try:
        env.reset(seed=SEED)
        env.command[:] = 0.0
        for _ in range(SETTLE_STEPS):
            env.step(np.zeros(12, dtype=np.float32))
        camera = PlannerCamera(env)
        rgb = camera.capture()
        image_path = output / "overhead-rgb.png"
        Image.fromarray(rgb).save(image_path)
        observation = env._obs().copy()
        yaw = heading(env)
        physical_energy, physical_held = float(env.energy), bool(env.held)
        scenarios = [
            ("eat-ball", "eat ball", physical_energy, physical_held, None),
            ("fetch-normal", "fetch the ball", physical_energy, physical_held, None),
            (
                "fetch-held-input",
                "fetch the ball",
                physical_energy,
                True,
                "Set only the model's held input to true; no physical grasp occurred.",
            ),
            (
                "fetch-low-energy-input",
                "fetch the ball",
                0.10,
                physical_held,
                "Set only the model's virtual-energy input to 0.10; the scene is unchanged.",
            ),
            ("behind", "go behind the ball", physical_energy, physical_held, None),
            ("push", "push the crate", physical_energy, physical_held, None),
            ("follow", "follow me", physical_energy, physical_held, None),
            ("novel-french", "teach me French", physical_energy, physical_held, None),
        ]
        examples = []
        for name, instruction, energy, held, intervention in scenarios:
            result = infer(model, rgb, instruction, observation, energy, float(held), yaw)
            examples.append(
                {
                    "id": name,
                    "instruction": instruction,
                    "model_input": {"energy": energy, "held": held, "yaw_rad": yaw},
                    "input_intervention": intervention,
                    "raw_decoded_prediction": result,
                }
            )
        sources = [
            "src/robopup/scene_planner.py",
            "src/robopup/planner_v3.py",
            "src/robopup/planner_v4.py",
            "src/robopup/brain.py",
            "src/robopup/executor.py",
            "src/robopup/env.py",
            "src/robopup/pet.py",
            "src/robopup/vision.py",
            "src/robopup/model.py",
            "benchmarks/task-flow-examples/export_examples.py",
        ]
        report = {
            "purpose": "Illustrative actual model predictions; not a held-out test or task success benchmark",
            "command": (
                "MUJOCO_GL=egl uv run --frozen python benchmarks/task-flow-examples/export_examples.py --out "
                + shlex.quote(str(args.out))
            ),
            "source_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=REPO,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip(),
            "source_sha256": {name: sha256(REPO / name) for name in sources},
            "model": {
                "version": checkpoint["version"],
                "checkpoint": "demo/arena-planner/planner.pt",
                "checkpoint_sha256": sha256(directory / "planner.pt"),
                "vocabulary_sha256": sha256(directory / "vocabulary.json"),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "inference_device": "cpu",
                "torch_version": str(torch.__version__),
                "torch_threads": torch.get_num_threads(),
                "new_training_performed": False,
            },
            "scene": {
                "seed": SEED,
                "reset_randomize": False,
                "settle_steps": SETTLE_STEPS,
                "physics_control_dt_s": env.dt,
                "settle_control_command": [0.0, 0.0],
                "settle_action": "12 zero residuals per step; analytic standing targets",
                "simulation_time_s": float(env.data.time),
                "after_render_physics_steps": 0,
                "physical_energy": physical_energy,
                "physical_held": physical_held,
                "motor_observation_50": observation.tolist(),
                "yaw_rad": yaw,
                "rgb": image_path.name,
                "rgb_width": int(rgb.shape[1]),
                "rgb_height": int(rgb.shape[0]),
                "rgb_sha256": sha256(image_path),
                "pixel_source": "Actual MuJoCo orthographic overhead camera; same frame for every example",
            },
            "interpretation": {
                "plan_durations": "duration_s=1.0 is fixed display metadata, not a duration prediction",
                "execution": "No commands executed for these examples; only standing-settle physics ran",
                "live_use": "GoalExecutor consults row 0; BrainSession replans every 0.5 simulated seconds",
                "confidence": "Model softmax scores are not calibrated correctness probabilities",
                "limits": "Arena colour classes, bounded task vocabulary and learned training aliases; not general language understanding",
            },
            "examples": examples,
        }
        (output / "examples.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "model": report["model"],
                    "scene_time_s": report["scene"]["simulation_time_s"],
                    "examples": [
                        {
                            "id": e["id"],
                            "intent": e["raw_decoded_prediction"]["intent"],
                            "rows": [
                                [r["skill"], r["object"]]
                                for r in e["raw_decoded_prediction"]["rows"]
                            ],
                        }
                        for e in examples
                    ],
                },
                indent=2,
            )
        )
    finally:
        if camera is not None:
            camera.close()
        env.close()


if __name__ == "__main__":
    main()
