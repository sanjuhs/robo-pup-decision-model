"""Fixed-seed evaluation. Gait and learned policies share an identical protocol."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from robopup.env import RoboPupEnv

EVAL_SEEDS = (1001, 1002, 1003, 1004, 1005)


def evaluate(policy=None, *, seeds=EVAL_SEEDS, target_speed=.35, episode_seconds=12):
    episodes = []
    for seed in seeds:
        env = RoboPupEnv(target_speed=target_speed, episode_seconds=episode_seconds)
        obs, _ = env.reset(seed=seed)
        total, speeds, energy = 0., [], []
        for step in range(env.max_steps):
            action = np.zeros(12) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, reward, done, truncated, info = env.step(action)
            total += reward
            speeds.append(info["speed"])
            energy.append(info["energy"])
            if done or truncated:
                break
        episodes.append({"seed": seed, "return": total, "steps": step+1,
                         "mean_speed_mps": float(np.mean(speeds)),
                         "mean_squared_torque": float(np.mean(energy)),
                         "distance_m": info["distance"], "fallen": info["fallen"],
                         "survival_fraction": (step+1)/env.max_steps})
        env.close()
    return {"protocol": "flat-forward-v1", "target_speed_mps": target_speed,
            "episode_seconds": episode_seconds, "episodes": episodes,
            "mean_return": float(np.mean([e["return"] for e in episodes])),
            "mean_speed_mps": float(np.mean([e["mean_speed_mps"] for e in episodes])),
            "mean_distance_m": float(np.mean([e["distance_m"] for e in episodes])),
            "fall_rate": float(np.mean([e["fallen"] for e in episodes])),
            "mean_survival_fraction": float(np.mean([e["survival_fraction"] for e in episodes]))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--speed", type=float, default=.35)
    args = p.parse_args()
    policy = PPO.load(args.model, device="cpu") if args.model else None
    result = evaluate(policy, target_speed=args.speed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
