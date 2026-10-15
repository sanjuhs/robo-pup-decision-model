"""Single-process local arena. Each WebSocket owns a separate physics world."""
from __future__ import annotations

import asyncio
import json
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robopup.decisions import decide, follow_target
from robopup.commands import interpret
from robopup.env import RoboPupEnv

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ROBOPUP_ARTIFACTS", ROOT / "artifacts")).resolve()
DIST = ROOT / "web" / "dist"
app = FastAPI(title="RoboPup Arena", version="0.1.0")
active_sessions = 0


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["manual", "approach", "follow", "push", "stop", "recover"] | None = None
    target: tuple[float, float] | None = None
    forward: float = Field(default=0., ge=-.4, le=.6)
    turn: float = Field(default=0., ge=-1.5, le=1.5)
    reset: bool = False
    paused: bool | None = None
    policy: str | None = None
    text: str | None = Field(default=None, max_length=160)


def runs():
    found = []
    if not ARTIFACTS.exists():
        return found
    for path in sorted(ARTIFACTS.glob("*/*/result.json")):
        if not path.resolve().is_relative_to(ARTIFACTS):
            continue
        try:
            result = json.loads(path.read_text())
            config = json.loads((path.parent / "config.json").read_text())
            checkpoint = path.parent / "policy.zip"
            if not checkpoint.resolve().is_relative_to(ARTIFACTS):
                continue
            evaluation = result["evaluation"]
            found.append({"id": str(path.parent.relative_to(ARTIFACTS)),
                          "participant": path.parent.parent.name,
                          "run": path.parent.name, "steps": result["actual_steps"],
                          "seconds": result["training_seconds"],
                          "speed": evaluation["mean_speed_mps"],
                          "return": evaluation["mean_return"],
                          "fall_rate": evaluation["fall_rate"],
                          "target_speed": config["target_speed"],
                          "has_policy": checkpoint.is_file()})
        except (OSError, KeyError, ValueError):
            continue
    return found


@app.get("/api/health")
def health():
    return {"status": "ok", "physics": "MuJoCo CPU", "version": "0.1.0",
            "sessions": active_sessions, "max_sessions": 12}


@app.get("/api/runs")
def list_runs():
    return {"runs": runs(), "protocol": "flat-forward-v1"}


@app.websocket("/ws")
async def arena(socket: WebSocket):
    global active_sessions
    origin = socket.headers.get("origin")
    if origin and urlsplit(origin).netloc != socket.headers.get("host"):
        await socket.close(code=1008)
        return
    if active_sessions >= 12:
        await socket.close(code=1013)
        return
    await socket.accept()
    active_sessions += 1
    env = RoboPupEnv(obstacles=True, randomize=False, episode_seconds=3600)
    obs, _ = env.reset(seed=7)
    mode, target, manual, paused = "stop", np.array([2., 1.]), (0., 0.), False
    policy, policy_name, last_control = None, "Analytic gait", 0.
    waypoints = []
    try:
        await socket.send_json({"type": "scene", "geoms": env.scene()})
        while True:
            tick = asyncio.get_running_loop().time()
            try:
                raw = await asyncio.wait_for(socket.receive_json(), timeout=.002)
                control = Control.model_validate(raw)
                if control.mode is not None:
                    mode = control.mode
                    waypoints = []
                if control.target is not None:
                    target = np.clip(control.target, -5, 5)
                    waypoints = []
                if control.text is not None:
                    command = interpret(control.text,
                        env.data.xpos[env.ball] if env.ball >= 0 else None, env.data.qpos[:3])
                    mode, waypoints = command.mode, command.waypoints
                    if waypoints:
                        target = np.asarray(waypoints.pop(0))
                    await socket.send_json({"type": "reply", "message": command.reply})
                manual = (control.forward, control.turn)
                last_control = tick
                if control.paused is not None:
                    paused = control.paused
                if control.reset:
                    obs, _ = env.reset(seed=7)
                    mode, manual = "stop", (0., 0.)
                    waypoints = []
                if control.policy is not None:
                    if control.policy == "gait":
                        policy, policy_name = None, "Analytic gait"
                    elif control.policy in {r["id"] for r in runs() if r["has_policy"]}:
                        from stable_baselines3 import PPO
                        import torch
                        torch.set_num_threads(1)
                        policy = PPO.load(ARTIFACTS / control.policy / "policy.zip", device="cpu")
                        policy_name = control.policy
                    else:
                        await socket.send_json({"type": "error", "message": "Unknown trusted policy"})
            except asyncio.TimeoutError:
                pass
            except ValidationError:
                await socket.send_json({"type": "error", "message": "Invalid controls"})
            except ValueError as error:
                await socket.send_json({"type": "error", "message": str(error)[:200]})
            except TypeError:
                await socket.send_json({"type": "error", "message": "Invalid controls"})
            if mode == "manual" and tick-last_control > .5:
                manual = (0., 0.)
            if mode == "follow":
                target = follow_target(env.data.time)
            if mode == "approach" and waypoints and np.linalg.norm(target-env.data.qpos[:2]) < .25:
                target = np.asarray(waypoints.pop(0))
            rotation = env.data.xmat[env.torso].reshape(3, 3)
            decision = decide(mode, env.data.qpos[:3], math.atan2(rotation[1, 0], rotation[0, 0]),
                              env.upright, target, env.data.xpos[env.crate], manual)
            env.command[:] = decision.command
            fallen = env.data.qpos[2] < .19 or env.upright < .35
            if not paused and not fallen:
                for _ in range(2):
                    # Recompute observations after a command changes.
                    obs = env._obs()
                    action = (np.zeros(12) if policy is None or np.linalg.norm(env.command)<.03
                              else policy.predict(obs, deterministic=True)[0])
                    obs, _, done, _, _ = env.step(action)
                    if done:
                        break
            state = env.snapshot()
            state.update({"type": "state", "behavior": decision.behavior,
                          "reason": decision.reason, "target": target.tolist(),
                          "paused": paused, "policy": policy_name,
                          "needs_reset": env.data.qpos[2] < .19 or env.upright < .35})
            await socket.send_json(state)
            await asyncio.sleep(max(0., .04-(asyncio.get_running_loop().time()-tick)))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        env.close()
        active_sessions -= 1


if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index():
    if (DIST / "index.html").exists():
        return FileResponse(DIST / "index.html")
    return {"message": "Build the browser first: cd web && npm ci && npm run build"}
