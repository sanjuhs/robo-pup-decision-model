"""Measure experimental behavior outcomes without teleporting or applying forces."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from robopup.decisions import decide, follow_target
from robopup.env import RoboPupEnv


def assess(mode, policy=None, seconds=30):
    env = RoboPupEnv(obstacles=True, randomize=False, episode_seconds=seconds)
    env.reset(seed=101)
    target = np.array([1., 0.])
    distances = []
    fallen = False
    for _ in range(env.max_steps):
        if mode == "follow":
            target = follow_target(env.data.time)
        rotation = env.data.xmat[env.torso].reshape(3, 3)
        decision = decide(mode, env.data.qpos[:3], math.atan2(rotation[1,0],rotation[0,0]),
                          env.upright, target, env.data.xpos[env.crate])
        env.command[:] = decision.command
        obs = env._obs()
        action = (np.zeros(12) if policy is None or np.linalg.norm(env.command)<.03
                  else policy.predict(obs, deterministic=True)[0])
        _, _, fallen, truncated, _ = env.step(action)
        distances.append(float(np.linalg.norm(target-env.data.qpos[:2])))
        if fallen or truncated:
            break
    crate_distance = float(np.linalg.norm(env.data.xpos[env.crate,:2]-np.array([3.,0.])))
    success = (not fallen and ((distances[-1]<.25) if mode=="approach" else
               (float(np.mean(distances[-250:]))<.5) if mode=="follow" else
               (crate_distance<.25) if mode=="push" else
               (np.linalg.norm(env.data.qpos[:2])<.15)))
    result = {"behavior":mode,"seconds":float(env.data.time),"fallen":fallen,
              "success":bool(success),"final_target_distance_m":distances[-1],
              "mean_target_distance_m":float(np.mean(distances)),
              "final_crate_to_goal_m":crate_distance,
              "robot_displacement_m":float(np.linalg.norm(env.data.qpos[:2]))}
    env.close()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    model=PPO.load(args.model,device="cpu") if args.model else None
    result={"policy":str(args.model) if args.model else "analytic gait",
            "protocol":"yard-behaviors-v1; one deterministic 30-second trial per behavior",
            "criteria":{"approach":"final distance <0.25m and no fall",
                        "follow":"mean distance over final 5s <0.5m and no fall",
                        "push":"crate within 0.25m of goal and no fall",
                        "stop":"displacement <0.15m and no fall"},
            "tasks":[assess(m,model) for m in ("stop","approach","follow","push")]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
