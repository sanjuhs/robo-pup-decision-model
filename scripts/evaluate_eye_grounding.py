#!/usr/bin/env python3
"""Offline eye grounding evaluation; simulator labels never enter live inference.

Run from a checkout with `uv run python scripts/evaluate_eye_grounding.py
--out artifacts/new-eye-probe`. Output directories must not already exist.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")  # Must precede the first MuJoCo import.

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import mujoco
import numpy as np
from PIL import Image
import torch

from robopup.eye_grounding import EyeGrounding, OBJECTS, require_visible_target
from robopup.pet import PetEnv
from robopup.vision import EyeCamera
from robopup.vlm_protocol import MissionStep

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATION = {
    "posterior_threshold": .9,
    "minimum_area_fraction": 1 / 2304,
    "minimum_pixels": 4,
    "fresh_positions": [[-.45, .2], [.55, -.3], [-.2, -.55]],
    "fresh_yaw_degrees": [-80, -25, 25, 105, 180],
    "seed_start": 1741,
    "occlusion_seed": 1839,
    "occlusion_robot_position": [0, 0],
    "occlusion_robot_yaw_degrees": 90,
    "occlusion_ball_positions": [[.3, .9, .125], [.3, 1.8, .125]],
    "thresholds_frozen_before_fresh_rendering": True,
}


class Evaluator:
    def __init__(self, out, model):
        self.out, self.model = out, model
        self.records = []

    def record(self, identifier, image, *, source, labels=None):
        began = time.perf_counter()
        report, mask = self.model.observe_with_mask(image)
        duration = (time.perf_counter() - began) * 1000
        Image.fromarray(image).save(self.out / f"{identifier}-rgb.png")
        indexed = Image.fromarray(mask).convert("P")
        palette = [0, 0, 0, 240, 100, 85, 255, 165, 45, 45, 105, 255, 195, 45, 225]
        indexed.putpalette(palette + [0] * (768 - len(palette)))
        indexed.save(self.out / f"{identifier}-learned-mask.png")
        rejection = None
        try:
            require_visible_target(MissionStep("fetch", "ball"), report)
        except ValueError as error:
            rejection = str(error)
        result = {
            "id": identifier, "source": source, "report": report,
            "fetch_guard_rejection": rejection,
            "rgb_sha256": hashlib.sha256(
                (self.out / f"{identifier}-rgb.png").read_bytes()
            ).hexdigest(),
            "inference_ms_cpu_one_thread": duration,
        }
        if labels is not None:
            result["offline_evaluation_only_simulator_pixel_counts"] = labels
        self.records.append(result)

    def actual_frame(self, identifier, env, camera):
        image = camera.capture()
        # Offline truth only: this evaluator owns simulator access. The model
        # receives only RGB; these counts never enter its inference inputs.
        camera.renderer.enable_segmentation_rendering()
        camera.renderer.update_scene(env.data, camera="robot_eyes")
        labels = camera.renderer.render().copy()
        camera.renderer.disable_segmentation_rendering()
        geom = labels[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        counts = {}
        names = ("ball_geom", "crate_geom", "home_pad", "beacon_geom")
        for name, geom_name in zip(OBJECTS, names, strict=True):
            geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            counts[name] = int(((labels[:, :, 0] == geom_id) & geom).sum())
        self.record(identifier, image, source="fresh MuJoCo head camera; fixed scene/pose recipe",
                    labels=counts)

    def recorded_frames(self, source):
        manifest = json.loads((source / "probe.json").read_text())
        for record in manifest["records"]:
            if not record["id"].startswith("gpu-"):
                continue
            filename = record["id"] + "-rgb.png"
            with Image.open(source / filename) as opened:
                image = np.asarray(opened.convert("RGB"))
            self.record(record["id"], image, source=record["source"])

    def fresh_frames(self):
        poses = ((position, yaw) for position in CONFIGURATION["fresh_positions"]
                 for yaw in CONFIGURATION["fresh_yaw_degrees"])
        for index, (position, yaw) in enumerate(poses):
            env = PetEnv()
            env.reset(seed=CONFIGURATION["seed_start"] + index)
            camera = EyeCamera(env)
            try:
                _settle(env)
                angle = math.radians(yaw)
                env.data.qpos[:2] = position
                env.data.qpos[3:7] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
                mujoco.mj_forward(env.model, env.data)
                self.actual_frame(f"fresh-{index:02d}", env, camera)
            finally:
                camera.close()
                env.close()
        identifiers = ("fresh-ball-before-occluder", "fresh-ball-behind-occluder")
        for identifier, position in zip(identifiers, CONFIGURATION["occlusion_ball_positions"], strict=True):
            env = PetEnv()
            env.reset(seed=CONFIGURATION["occlusion_seed"])
            camera = EyeCamera(env)
            try:
                _settle(env)
                env.data.qpos[:2] = CONFIGURATION["occlusion_robot_position"]
                env.data.qpos[3:7] = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
                joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
                address = env.model.jnt_qposadr[joint]
                env.data.qpos[address:address + 3] = position
                mujoco.mj_forward(env.model, env.data)
                self.actual_frame(identifier, env, camera)
            finally:
                camera.close()
                env.close()

    def result(self):
        fresh = [record for record in self.records
                 if "offline_evaluation_only_simulator_pixel_counts" in record]
        summary = {
            name: {"visible_truth_min64": 0, "true_positive": 0, "false_positive": 0,
                   "false_negative": 0, "true_negative": 0}
            for name in OBJECTS
        }
        for row in fresh:
            for name in OBJECTS:
                truth = row["offline_evaluation_only_simulator_pixel_counts"][name] >= 64
                prediction = row["report"]["objects"][name]["visible"]
                summary[name]["visible_truth_min64"] += int(truth)
                outcome = (("true_positive" if prediction else "false_negative") if truth
                           else ("false_positive" if prediction else "true_negative"))
                summary[name][outcome] += 1
        return {
            "configuration": CONFIGURATION,
            "checkpoint_sha256": self.model.checkpoint_sha256,
            "implementation_sha256": hashlib.sha256(
                (ROOT / "src/robopup/eye_grounding.py").read_bytes()
            ).hexdigest(),
            "evaluation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runtime": {"torch": torch.__version__, "device": "cpu", "threads": 1,
                        "platform": platform.platform(), "mujoco_gl": os.environ["MUJOCO_GL"]},
            "trained_pixel_parameters": 821,
            "new_training_performed": False,
            "model_input": "RGB image only; no coordinates, IDs, segmentation or target truth",
            "evaluation_labels": "Renderer IDs only count true object pixels offline in fresh frames",
            "fresh_cases": len(fresh), "recorded_gpu_cases": len(self.records) - len(fresh),
            "fresh_summary_at_ground_truth_min64_pixels": summary,
            "limitations": [
                "18 positive and 50 negative same-colour/renderer checks, not generalization evidence",
                "Posterior/area thresholds are engineering abstention rules, not probability calibration",
                "No grounding can mean an occluded, small, shadowed or out-of-domain object",
                "Visibility does not establish task completion or prevent repeats of a visible target",
            ],
            "records": self.records,
        }


def _settle(env):
    env.command[:] = 0
    for _ in range(25):
        env.step(np.zeros(12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="New output directory; existing directories are never overwritten")
    parser.add_argument("--model", type=Path, default=ROOT / "demo/arena-planner")
    parser.add_argument("--recorded-source", type=Path,
                        default=ROOT / "benchmarks/vlm-2026-10-16/eye-grounding")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output directory already exists; choose a new --out")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    (args.out / "frozen-probe-configuration.json").write_text(
        json.dumps(CONFIGURATION, indent=2) + "\n"
    )
    evaluator = Evaluator(args.out, EyeGrounding(args.model))
    evaluator.recorded_frames(args.recorded_source)
    evaluator.fresh_frames()
    result = evaluator.result()
    (args.out / "probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"fresh_cases": result["fresh_cases"],
                      "recorded_gpu_cases": result["recorded_gpu_cases"],
                      "summary": result["fresh_summary_at_ground_truth_min64_pixels"]}, indent=2))


if __name__ == "__main__":
    main()
