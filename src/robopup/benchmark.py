"""Measure actual native simulation throughput; never infer GPU performance."""
import argparse
import json
import platform
from pathlib import Path
import time

import mujoco
import numpy as np

from robopup.env import RoboPupEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    env = RoboPupEnv()
    env.reset(seed=41)
    action = np.zeros(12)
    resets = 0
    start = time.perf_counter()
    for _ in range(args.steps):
        _, _, done, truncated, _ = env.step(action)
        if done or truncated:
            env.reset(seed=41+resets)
            resets += 1
    elapsed = time.perf_counter()-start
    result = {"physics_device": "cpu", "python": platform.python_version(),
              "mujoco": mujoco.__version__, "steps": args.steps, "seconds": elapsed,
              "steps_per_second": args.steps/elapsed, "resets": resets,
              "simulated_seconds": args.steps*env.dt, "control_hz": 50,
              "physics_hz": 200, "paid_gpu_spend_usd": 0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
