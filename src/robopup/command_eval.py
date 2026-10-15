"""Physical mission outcomes, separate from parser/acknowledgment tests."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from robopup.commands import interpret
from robopup.decisions import decide
from robopup.env import RoboPupEnv


def trial(text, policy=None):
    env = RoboPupEnv(obstacles=True, randomize=False, episode_seconds=60)
    env.reset(seed=101)
    command = interpret(text, env.data.xpos[env.ball], env.data.qpos[:3])
    points = list(command.waypoints)
    target = np.asarray(points.pop(0))
    success, fallen = False, False
    for _ in range(env.max_steps):
        if points and np.linalg.norm(target-env.data.qpos[:2]) < .25:
            target = np.asarray(points.pop(0))
        rotation = env.data.xmat[env.torso].reshape(3,3)
        decision = decide(command.mode, env.data.qpos[:3],
            math.atan2(rotation[1,0],rotation[0,0]), env.upright, target, env.data.xpos[env.crate])
        env.command[:] = decision.command
        action = (np.zeros(12) if policy is None or np.linalg.norm(env.command)<.03
                  else policy.predict(env._obs(), deterministic=True)[0])
        _, _, fallen, truncated, _ = env.step(action)
        ball, pos = env.data.xpos[env.ball], env.data.qpos[:3]
        success = (env.ball_touched if command.goal=="ball" else
                   not points and pos[0]>ball[0]+.35 and abs(pos[1]-ball[1])<.45)
        if fallen or truncated or success:
            break
    result = {"command":text, "seconds":float(env.data.time), "success":bool(success and not fallen),
              "fallen":fallen, "ball_contact":env.ball_touched,
              "final_ball_distance_m":float(np.linalg.norm(env.data.qpos[:2]-env.data.xpos[env.ball,:2])),
              "remaining_waypoints":len(points), "robot_xyz":env.data.qpos[:3].tolist(),
              "ball_xyz":env.data.xpos[env.ball].tolist()}
    env.close()
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    policy=PPO.load(args.model,device="cpu") if args.model else None
    result={"protocol":"ball-missions-v2; one deterministic trial, maximum 60s per command",
            "policy":str(args.model) if args.model else "analytic gait",
            "criteria":{"go to ball":"MuJoCo contact between robot and ball, no fall",
                        "go behind ball":"all waypoints used; robot on +X side by >0.35m, Y within 0.45m; no fall"},
            "trials":[trial(text,policy) for text in ("go to the ball","go behind the ball")]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
