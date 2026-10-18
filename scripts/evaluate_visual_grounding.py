#!/usr/bin/env python3
"""Paired visual counterfactuals, with independent raw-VLM and gate scoring.

Generate frozen scenes and run the cheap guard with --out DIR. Add --run-model
for genuine VLM inference via the existing runtime (no automatic download).
--replay-original scores archived genuine Qwen outputs, never new inference.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import subprocess
import time

import mujoco
from PIL import Image
import torch

from evaluate_eye_grounding import CONFIGURATION, Evaluator, _settle
from robopup.eye_grounding import EyeGrounding
from robopup.visual_grounding_eval import decision, score_guard, score_pairs, summarize_guard
from robopup.vlm_protocol import MissionValidationError
from robopup.vlm_runtime import VLMRuntime, VLMRuntimeError
from robopup.pet import PetEnv
from robopup.vision import EyeCamera

ROOT = Path(__file__).resolve().parents[1]
PRESENCE = ("Inspect the image. If you see a red ball, fetch it then stop. "
            "If you cannot see a red ball, look then stop.")
POSITION = ("Inspect the image. If a red ball is visible on the left half of the image, "
            "turn left then stop. If it is on the right half, turn right then stop. "
            "If you cannot see a red ball, look then stop. Do not fetch it.")
# Frozen before rendering/scoring; no threshold tuning from these evaluation views.
RECIPES = [
    {"id": "missing", "category": "missing_object", "task": "presence",
     "variants": [{"ball": [1.8, -.7, .125]}, {"ball": [1.8, -.7, -5]}]},
    {"id": "position", "category": "changed_position", "task": "position",
     "variants": [{"ball": [1.8, .7, .125]}, {"ball": [1.8, -.7, .125]}]},
    {"id": "occlusion", "category": "occlusion", "task": "presence",
     "variants": [{"yaw": 90, "ball": position} for position in
                  CONFIGURATION["occlusion_ball_positions"]]},
    {"id": "novel-view", "category": "unfamiliar_viewpoint", "task": "position",
     "variants": [{"yaw": 147, "robot": [-.8, -.4], "pitch": -12,
                   "relative_ball": [1.8, side, .125]} for side in (.7, -.7)]},
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def render_pairs(out):
    evaluator = Evaluator(out, EyeGrounding())
    pairs = []
    for recipe in RECIPES:
        pair = {key: recipe[key] for key in ("id", "category", "task")}
        pair.update(instruction=PRESENCE if recipe["task"] == "presence" else POSITION,
                    energy=.9, held=False, cases=[])
        for index, variant in enumerate(recipe["variants"]):
            env = PetEnv()
            env.reset(seed=CONFIGURATION["occlusion_seed"])
            camera = EyeCamera(env)
            try:
                _settle(env)
                robot = variant.get("robot", [0, 0])
                yaw = math.radians(variant.get("yaw", 0))
                pitch = math.radians(variant.get("pitch", 0))
                env.data.qpos[:2] = robot
                env.data.qpos[3:7] = [math.cos(yaw / 2) * math.cos(pitch / 2),
                                    -math.sin(yaw / 2) * math.sin(pitch / 2),
                                    math.cos(yaw / 2) * math.sin(pitch / 2),
                                    math.sin(yaw / 2) * math.cos(pitch / 2)]
                ball = variant.get("ball")
                if ball is None:
                    x, y, z = variant["relative_ball"]
                    ball = [robot[0] + x * math.cos(yaw) - y * math.sin(yaw),
                            robot[1] + x * math.sin(yaw) + y * math.cos(yaw), z]
                joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
                address = env.model.jnt_qposadr[joint]
                env.data.qpos[address:address + 3] = ball
                mujoco.mj_forward(env.model, env.data)
                identifier = f"{recipe['id']}-{index}"
                evaluator.actual_frame(identifier, env, camera)
                record = evaluator.records[-1]
                pixels = record["offline_evaluation_only_simulator_pixel_counts"]["ball"]
                centroid = record["offline_evaluation_only_horizontal_centroids"]["ball"]
                visible = pixels >= 64
                expected = ("fetch" if visible else "look")
                if recipe["task"] == "position":
                    if not visible or centroid is None or abs(centroid - .5) < .1:
                        raise ValueError("Position recipe failed offline visibility/side validation")
                    expected = "turn_left" if centroid < .5 else "turn_right"
                case = {"id": identifier, "image": f"{identifier}-rgb.png",
                        "sha256": record["rgb_sha256"], "expected_decision": expected,
                        "offline_truth_ball_pixels": pixels, "offline_truth_ball_x": centroid,
                        "guard": score_guard(record["report"], truth_visible=visible),
                        "guard_report": record["report"],
                        "guard_ms": record["inference_ms_cpu_one_thread"]}
                pair["cases"].append(case)
            finally:
                camera.close()
                env.close()
        if pair["cases"][0]["expected_decision"] == pair["cases"][1]["expected_decision"]:
            raise ValueError(f"Counterfactual {pair['id']} did not change truth")
        if pair["cases"][0]["sha256"] == pair["cases"][1]["sha256"]:
            raise ValueError("Counterfactual images are identical")
        pairs.append(pair)
    return pairs


def run_model(pairs, out, runtime):
    records = []
    for pair in pairs:
        for case in pair["cases"]:
            image_path = out / case["image"]
            if hashlib.sha256(image_path.read_bytes()).hexdigest() != case["sha256"]:
                raise ValueError("Frozen image checksum mismatch")
            row = {"id": case["id"], "image_sha256": case["sha256"],
                   "actual_inference_attempted": True}
            began = time.perf_counter()
            try:
                # Only image, identical instruction, and fixed state enter the runtime.
                with Image.open(image_path) as image:
                    generation = runtime.generate(image, pair["instruction"],
                                                  pair["energy"], pair["held"])
                row["generation"] = generation.as_dict()
                row["decision"] = decision(generation.raw_text, pair["task"])
            except (VLMRuntimeError, MissionValidationError) as exc:
                row["error"] = str(exc)
            row["wall_ms"] = (time.perf_counter() - began) * 1000
            records.append(row)
            write_json(out / f"{case['id']}-generation.json", row)
            # Preserve partial progress and errors on interruption.
            write_json(out / "model-results.json", {"records": records,
                       "summary": score_pairs(pairs, records), "runtime": runtime.status()})
    return records



def rescore_recorded_run(pairs, source):
    """Re-score raw generations only when all paired inputs match exactly."""
    original = json.loads((source / "pairs.json").read_text())
    def inputs(items):
        return [(pair["id"], pair["task"], pair["instruction"], pair["energy"], pair["held"],
                 [(case["id"], case["sha256"]) for case in pair["cases"]]) for pair in items]
    if inputs(pairs) != inputs(original):
        raise ValueError("Recorded run inputs do not match frozen paired inputs")
    records = []
    for pair in pairs:
        for case in pair["cases"]:
            path = source / f"{case['id']}-generation.json"
            if not path.is_file():
                records.append({"id": case["id"], "error": "Missing recorded generation"})
                continue
            row = json.loads(path.read_text())
            if row["id"] != case["id"] or row["image_sha256"] != case["sha256"]:
                raise ValueError("Recorded generation image/id mismatch")
            row.pop("decision", None)
            if "generation" in row:
                try:
                    row["decision"] = decision(row["generation"]["raw_text"], pair["task"])
                except MissionValidationError as exc:
                    row["error"] = str(exc)
            row["source_record_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            records.append(row)
    return {"new_inference": False, "source": "re-scored recorded runtime generations",
            "records": records, "summary": score_pairs(pairs, records)}


def replay_original():
    source = ROOT / "benchmarks/vlm-2026-10-16/gpu-4090"
    rows = []
    for name in ("case-002-visual-choice", "case-003-visual-choice-blank",
                 "case-004-visual-choice-turned"):
        path = source / f"{name}.json"
        data = json.loads(path.read_text())
        image = source / "inputs" / data["image"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != data["image_sha256"]:
            raise ValueError("Archived image checksum mismatch")
        if data["instruction"] != PRESENCE or not data["actual_generated_response_received"]:
            raise ValueError("Archive does not contain the matched genuine prompt")
        rows.append({"id": data["id"], "decision": decision(data["generation"]["raw_text"],
                                                             "presence"),
                     "source": str(path.relative_to(ROOT)),
                     "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    pairs = [{"id": f"archive-{index}", "category": "archived_image_ablation",
              "cases": [{"id": rows[0]["id"], "expected_decision": "fetch"},
                        {"id": rows[index]["id"], "expected_decision": "look"}]}
             for index in (1, 2)]
    return {"new_inference": False, "unique_images": 3,
            "note": "Two pairs share a baseline; not independent samples or a new model run",
            "records": rows, "summary": score_pairs(pairs, rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    inference = parser.add_mutually_exclusive_group()
    inference.add_argument("--run-model", action="store_true")
    inference.add_argument("--replay-model-results", type=Path,
                           help="Re-score an existing raw runtime run, with no new inference")
    parser.add_argument("--dataset", type=Path,
                        help="Reuse frozen pairs/images from an earlier run; do not render again")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--replay-original", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output directory; existing evidence is never overwritten")
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    write_json(args.out / "configuration.json", {"recipes": RECIPES, "truth_min_pixels": 64,
               "guard_thresholds": {key: CONFIGURATION[key] for key in
                                    ("posterior_threshold", "minimum_area_fraction",
                                     "minimum_pixels")},
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                        cwd=ROOT, text=True).strip(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "scorer_sha256": hashlib.sha256(
                   (ROOT / "src/robopup/visual_grounding_eval.py").read_bytes()).hexdigest(),
               "torch": torch.__version__, "mujoco": mujoco.__version__,
               "mujoco_gl": os.environ["MUJOCO_GL"], "threads": 1})
    if args.dataset:
        pairs = json.loads((args.dataset / "pairs.json").read_text())
        for pair in pairs:
            for case in pair["cases"]:
                name = case["image"]
                if Path(name).name != name:
                    raise ValueError("Dataset image must be a basename")
                source = args.dataset / name
                if hashlib.sha256(source.read_bytes()).hexdigest() != case["sha256"]:
                    raise ValueError("Dataset image checksum mismatch")
                shutil.copyfile(source, args.out / name)
        shutil.copyfile(args.dataset / "configuration.json", args.out / "dataset-configuration.json")
    else:
        pairs = render_pairs(args.out)
    write_json(args.out / "pairs.json", pairs)
    write_json(args.out / "guard-results.json", {
        "source": "existing frozen pixel gate, fixed fetch candidates, held=False",
        "reused_observations_from_dataset": bool(args.dataset),
        "summary": summarize_guard([case for pair in pairs for case in pair["cases"]]),
        "semantic_absence_rejections": 0,
        "note": "reject_ungrounded is abstention, not proof of absence; no action executed"})
    if args.replay_original:
        write_json(args.out / "archived-model-results.json", replay_original())
    if args.run_model:
        runtime = VLMRuntime(model_dir=args.model_dir)
        write_json(args.out / "runtime-before.json", runtime.status())
        run_model(pairs, args.out, runtime)
    elif args.replay_model_results:
        write_json(args.out / "model-results.json", rescore_recorded_run(
            pairs, args.replay_model_results))
    else:
        write_json(args.out / "model-results.json", {"executed": False,
                   "reason": "New model inference requires --run-model and available weights/provider"})
    print(json.dumps({"out": str(args.out), "pairs": len(pairs),
                      "new_model_inference_requested": args.run_model}))


if __name__ == "__main__":
    main()
