#!/usr/bin/env python3
"""Inspect bundled trusted checkpoints; never downloads models or starts training."""

import argparse
import hashlib
import json
from pathlib import Path

from stable_baselines3 import PPO
import torch

from robopup.planner import MultimodalPlanner
from robopup.scene_planner import load_planner

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"checkpoints": [], "hosting": "GitHub repository; no Hugging Face upload"}
    for path in sorted((ROOT / "demo").glob("*/policy.zip")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != (path.parent / "SHA256SUMS").read_text().split()[0]:
            raise ValueError("Bundled checkpoint checksum mismatch: " + path.parent.name)
        policy = PPO.load(path, device="cpu").policy
        actor = [*policy.mlp_extractor.policy_net.parameters(), *policy.action_net.parameters()]
        report["checkpoints"].append(
            {
                "path": str(path.relative_to(ROOT)),
                "checkpoint_bytes": path.stat().st_size,
                "sha256": digest,
                "policy_parameters": sum(p.numel() for p in policy.parameters()),
                "policy_parameter_bytes": sum(
                    p.numel() * p.element_size() for p in policy.parameters()
                ),
                "deterministic_actor_parameters": sum(p.numel() for p in actor),
                "deterministic_actor_parameter_bytes": sum(
                    p.numel() * p.element_size() for p in actor
                ),
                "observations": 50,
                "joint_residual_outputs": 12,
                "architecture": "separate actor/critic 50->64->64 MLP; actor->12, critic->1",
            }
        )
    path = ROOT / "demo/arena-planner/planner.pt"
    trained, _ = load_planner(path.parent)
    report["trained_arena_planner"] = {
        "path": str(path.relative_to(ROOT)),
        "checkpoint_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "parameters": sum(p.numel() for p in trained.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in trained.parameters()),
        "inputs": "96x96 overhead RGB + up to 96 word tokens + 53 numerical values",
        "outputs": "six skill/object/offset proposals; fixed one-second chunks",
        "training": "supervised arena imitation, not task RL",
        "status": "bundled trained weights, default live decision brain; bounded arena language/vision",
    }
    planner = MultimodalPlanner()
    report["planner_prototype"] = {
        "parameters": sum(p.numel() for p in planner.parameters()),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in planner.parameters()),
        "status": "Random initialization; no trained checkpoint; not connected to arena",
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
