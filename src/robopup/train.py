"""Fresh PPO residual policies, exclusive output directories, wall-time guard."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from robopup.env import RoboPupEnv
from robopup.evaluate import evaluate
from robopup.experiments import TrainConfig, participant_path


class Deadline(BaseCallback):
    def __init__(self, seconds):
        super().__init__()
        self.seconds = seconds

    def _on_training_start(self):
        self.start = time.monotonic()

    def _on_step(self):
        return time.monotonic()-self.start < self.seconds


def run_training(config: TrainConfig, output: Path, max_minutes=20):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config_path = output / "config.json"
    config_path.write_text(config.model_dump_json(indent=2)+"\n")
    revision = subprocess.run(["git", "rev-parse", "HEAD"],capture_output=True,text=True)
    metadata = {"started_at": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
                "revision": revision.stdout.strip(), "policy": "residual-analytic-trot-v1",
                "training_device": config.device, "physics_device": "cpu",
                "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")},
                "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"],
                                  capture_output=True,text=True).stdout)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    env = make_vec_env(RoboPupEnv, n_envs=config.n_envs, seed=config.seed,
                       env_kwargs={"target_speed": config.target_speed,
                                   "energy_weight": config.energy_weight},
                       monitor_dir=str(output / "monitor"))
    model = PPO("MlpPolicy", env, seed=config.seed, device=config.device,
                learning_rate=config.learning_rate, n_steps=256, batch_size=256,
                n_epochs=5, policy_kwargs={"net_arch": [64, 64],
                                          "log_std_init": config.initial_log_std}, verbose=0)
    model.set_logger(configure(str(output), ["csv"]))
    callbacks = CallbackList([
        CheckpointCallback(save_freq=max(1, 32768//config.n_envs),
                           save_path=str(output / "checkpoints"), name_prefix="pup"),
        Deadline(max_minutes*60),
    ])
    start = time.monotonic()
    try:
        model.learn(total_timesteps=config.total_steps, callback=callbacks)
        elapsed = time.monotonic()-start
        model.save(output / "policy")
        summary = {"requested_steps": config.total_steps, "actual_steps": model.num_timesteps,
                   "training_seconds": elapsed, "steps_per_second": model.num_timesteps/elapsed,
                   "deadline_reached": model.num_timesteps < config.total_steps,
                   "evaluation": evaluate(model, target_speed=config.target_speed)}
        (output / "result.json").write_text(json.dumps(summary, indent=2)+"\n")
        print(json.dumps({k: v for k,v in summary.items() if k != "evaluation"}, indent=2))
        return summary
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/curious.json"))
    p.add_argument("--participant", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--steps", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--max-minutes", type=float, default=20)
    p.add_argument("--output-root", type=Path, default=Path("artifacts"))
    args = p.parse_args()
    if not 0 < args.max_minutes <= 120:
        p.error("--max-minutes must be in (0, 120]")
    data = json.loads(args.config.read_text())
    for key in ("seed", "device"):
        if getattr(args, key) is not None:
            data[key] = getattr(args, key)
    if args.steps is not None:
        data["total_steps"] = args.steps
    config = TrainConfig.model_validate(data)
    if config.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable; choose --device cpu")
    run_training(config, participant_path(args.output_root, args.participant, args.run),
                 args.max_minutes)


if __name__ == "__main__":
    main()
