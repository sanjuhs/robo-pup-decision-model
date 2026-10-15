"""One explicit MJX training run; compiled physics device is verified first."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

import jax
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import VecMonitor

from robopup.evaluate import evaluate
from robopup.mjx_env import MJXVecEnv
from robopup.train import Deadline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=1048576)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-minutes", type=float, default=10)
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--probe-only", action="store_true")
    args = p.parse_args()
    if not 1 <= args.envs <= 2048 or args.steps < 1024 or not 0 < args.max_minutes <= 30:
        p.error("Invalid environment count, step budget, or time limit")
    devices = [str(device) for device in jax.devices()]
    if args.require_gpu and not any(device.platform == "gpu" for device in jax.devices()):
        p.error("GPU physics requested, but JAX did not find a GPU")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = {"seed": args.seed, "total_steps": args.steps, "n_envs": args.envs,
              "target_speed": .35, "energy_weight": .002, "backend": "mjx",
              "device": "cpu", "initial_log_std": -2., "n_steps": 64,
              "physics_friction": .9, "solver": "CG/10/4"}
    (args.out / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "jax": jax.__version__, "jax_devices": devices, "torch": torch.__version__,
                "physics_device": jax.devices()[0].platform, "training_device": "cpu",
                "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in Path(__file__).parent.glob("*.py")}}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"stage": "compile", "devices": devices}), flush=True)
    compile_start = time.monotonic()
    vector = MJXVecEnv(args.envs, seed=args.seed)
    obs = vector.reset()
    vector.step(np.zeros((args.envs,12), dtype=np.float32))
    compile_seconds = time.monotonic()-compile_start
    assert obs.shape == (args.envs,50) and np.isfinite(obs).all()
    bench_start = time.monotonic()
    for _ in range(50):
        obs, rewards, dones, _ = vector.step(np.zeros((args.envs,12), dtype=np.float32))
        if not np.isfinite(obs).all() or not np.isfinite(rewards).all():
            raise RuntimeError("MJX probe produced non-finite results")
    benchmark = {"compile_seconds": compile_seconds,
                 "physics_device": metadata["physics_device"],
                 "batched_control_steps": 50*args.envs,
                 "batched_steps_per_second": 50*args.envs/(time.monotonic()-bench_start),
                 "envs": args.envs, "physics_hz": 200, "control_hz": 50}
    (args.out / "physics.json").write_text(json.dumps(benchmark, indent=2)+"\n")
    vector.state = (*vector.state[:3], vector.state[3]*0+vector.max_steps-1, vector.state[4])
    reset_obs, _, done, info = vector.step(np.zeros((args.envs,12), dtype=np.float32))
    assert done.all() and np.isfinite(reset_obs).all()
    assert all(item.get("terminal_observation") is not None for item in info)
    assert np.asarray(vector.state[3]).max() == 0
    vector.reset()
    print(json.dumps({"stage": "physics_probe", **benchmark}), flush=True)
    if args.probe_only:
        return
    env = VecMonitor(vector, filename=str(args.out / "monitor.csv"))
    policy = PPO("MlpPolicy", env, seed=args.seed, device="cpu", n_steps=64, batch_size=1024,
                 n_epochs=5, policy_kwargs={"net_arch": [64,64], "log_std_init": -2.}, verbose=0)
    policy.set_logger(configure(str(args.out), ["csv"]))
    start = time.monotonic()
    callbacks = CallbackList([Deadline(args.max_minutes*60),
        CheckpointCallback(save_freq=max(1, 131072//args.envs),
                           save_path=str(args.out / "checkpoints"), name_prefix="pup")])
    policy.learn(total_timesteps=args.steps, callback=callbacks)
    elapsed = time.monotonic()-start
    policy.save(args.out / "policy")
    result = {"actual_steps": policy.num_timesteps, "requested_steps": args.steps,
              "training_seconds": elapsed, "steps_per_second": policy.num_timesteps/elapsed,
              "deadline_reached": policy.num_timesteps < args.steps,
              "compile_seconds": compile_seconds, "evaluation": evaluate(policy)}
    (args.out / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"stage": "complete", "actual_steps": policy.num_timesteps,
                      "training_seconds": elapsed,
                      "mean_speed_mps": result["evaluation"]["mean_speed_mps"],
                      "fall_rate": result["evaluation"]["fall_rate"]}), flush=True)


if __name__ == "__main__":
    main()
