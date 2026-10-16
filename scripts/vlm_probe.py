#!/usr/bin/env python3
"""Run actual VLM image inference and validate the generated mission JSON.

An image file is mandatory. Use a robot-head camera capture for a robot-vision
claim. Successful JSON generation alone never proves a physical task succeeded.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from PIL import Image

from robopup.vlm_protocol import MissionValidationError, parse_mission
from robopup.vlm_runtime import VLMRuntime, VLMRuntimeError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--energy", type=float, default=0.8)
    parser.add_argument("--held", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    runtime = VLMRuntime(model_dir=args.model_dir, max_new_tokens=args.max_new_tokens)
    record = {
        "schema": "robopup-real-vlm-probe-v1", "timestamp_unix": time.time(),
        "instruction": args.instruction, "energy": args.energy, "held": args.held,
        "image": {"file": args.image.name,
                  "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest()},
        "initial_runtime_status": runtime.status(),
        "physical_task_tested": False, "valid_mission": False,
    }
    try:
        with Image.open(args.image) as image:
            generation = runtime.generate(image, args.instruction, args.energy, args.held)
        record["generation"] = generation.as_dict()
        record["mission"] = parse_mission(generation.raw_text).as_dict()
        record["valid_mission"] = True
    except (VLMRuntimeError, MissionValidationError) as exc:
        record["error"] = str(exc)
    record["final_runtime_status"] = runtime.status()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "valid_mission": record["valid_mission"],
                      "phase": record["final_runtime_status"]["phase"]}))
    if not record["valid_mission"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
