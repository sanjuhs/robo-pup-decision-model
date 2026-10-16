"""Single-process local arena. Each WebSocket owns a separate physics world."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hmac
from functools import lru_cache
import json
import math
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robopup.decisions import decide, follow_target
from robopup.commands import interpret
from robopup.pet import PetEnv
from robopup.brain import MODEL_DIR, trained_model
from robopup.vlm_mission import BrainRouter
from robopup.vlm_runtime import get_vlm_runtime
from robopup.vlm_api import router as vlm_api_router

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(os.environ.get("ROBOPUP_ARTIFACTS", ROOT / "artifacts")).resolve()
DIST = ROOT / "web" / "dist"
@asynccontextmanager
async def lifespan(_app):
    # Readiness means bundled inference is warm, not merely that the port opened.
    if (MODEL_DIR / "planner.pt").is_file():
        await asyncio.to_thread(default_motor)
        await asyncio.to_thread(trained_model)
    yield


app = FastAPI(title="RoboPup Arena", version="0.1.0", lifespan=lifespan)
app.include_router(vlm_api_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[s.strip() for s in os.getenv("ROBOPUP_ALLOWED_ORIGINS", "").split(",") if s.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
active_sessions = 0


def origin_allowed(origin, host):
    if not origin or urlsplit(origin).netloc == host:
        return True
    allowed = {s.strip().rstrip("/") for s in os.getenv("ROBOPUP_ALLOWED_ORIGINS", "").split(",")}
    # An externally hosted UI must use a private workshop access key.
    return bool(os.getenv("ROBOPUP_ARENA_TOKEN")) and origin.rstrip("/") in allowed


@app.middleware("http")
async def private_api(request: Request, call_next):
    token = os.getenv("ROBOPUP_ARENA_TOKEN", "")
    if token and request.url.path.startswith("/api/") and request.url.path not in ("/api/health", "/api/vlm") and request.method != "OPTIONS":
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            return JSONResponse({"detail": "Workshop access key required"}, status_code=401)
    return await call_next(request)


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    mode: Literal["manual", "approach", "follow", "push", "stop", "recover"] | None = None
    target: tuple[float, float] | None = None
    forward: float = Field(default=0.0, ge=-0.4, le=0.6)
    turn: float = Field(default=0.0, ge=-1.5, le=1.5)
    reset: bool = False
    paused: bool | None = None
    policy: str | None = None
    text: str | None = Field(default=None, max_length=2000)
    brain: Literal["neural", "rules", "vlm"] | None = None
    gait: Literal["walk", "trot", "run"] | None = None
    energy: float | None = Field(default=None, ge=0, le=1)
    release: bool = False


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
            found.append(
                {
                    "id": str(path.parent.relative_to(ARTIFACTS)),
                    "participant": path.parent.parent.name,
                    "run": path.parent.name,
                    "steps": result["actual_steps"],
                    "seconds": result["training_seconds"],
                    "speed": evaluation["mean_speed_mps"],
                    "return": evaluation["mean_return"],
                    "fall_rate": evaluation["fall_rate"],
                    "target_speed": config["target_speed"],
                    "has_policy": checkpoint.is_file(),
                }
            )
        except (OSError, KeyError, ValueError):
            continue
    return found


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "physics": "MuJoCo CPU",
        "version": "0.1.0",
        "sessions": active_sessions,
        "max_sessions": 12,
        "planner_available": (MODEL_DIR / "planner.pt").is_file(),
    }


@app.get("/api/runs")
def list_runs():
    return {"runs": runs(), "protocol": "flat-forward-v1"}


@app.get("/api/vlm")
def vlm_status():
    return get_vlm_runtime().status()


voice_slots = asyncio.Semaphore(2)


@app.post("/api/transcribe")
async def transcribe(request: Request):
    origin = request.headers.get("origin")
    if not origin_allowed(origin, request.headers.get("host")):
        raise HTTPException(403, "Cross-origin audio rejected")
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > 300_000:
            raise HTTPException(413, "Record eight seconds or less")
    from robopup.speech import transcribe_wav

    try:
        async with voice_slots:
            return await asyncio.to_thread(transcribe_wav, bytes(content))
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@lru_cache(maxsize=1)
def default_motor():
    from stable_baselines3 import PPO
    import torch

    torch.set_num_threads(1)
    return PPO.load(ROOT / "demo/curious-v2/policy.zip", device="cpu")


@app.websocket("/ws")
async def arena(socket: WebSocket):
    global active_sessions
    origin = socket.headers.get("origin")
    if not origin_allowed(origin, socket.headers.get("host")):
        await socket.close(code=1008)
        return
    if active_sessions >= 12:
        await socket.close(code=1013)
        return
    await socket.accept()
    # Flush the handshake before synchronous MuJoCo session construction.
    await asyncio.sleep(0)
    token = os.getenv("ROBOPUP_ARENA_TOKEN", "")
    if token:
        try:
            auth = await asyncio.wait_for(socket.receive_json(), timeout=5)
            if not isinstance(auth, dict) or auth.get("type") != "auth" or not isinstance(auth.get("token"), str) or not hmac.compare_digest(auth["token"].encode(), token.encode()):
                await socket.close(code=1008)
                return
        except (asyncio.TimeoutError, ValueError, WebSocketDisconnect):
            await socket.close(code=1008)
            return
    active_sessions += 1
    env = PetEnv(episode_seconds=3600)
    obs, _ = env.reset(seed=7)
    mode, target, manual, paused = "stop", np.array([2.0, 1.0]), (0.0, 0.0), False
    policy, policy_name, last_control = None, "Analytic gait", 0.0
    waypoints = []
    mission = None
    mission_complete = False
    brain = BrainRouter(env)
    prefer_vlm = os.getenv("ROBOPUP_DEFAULT_BRAIN") == "vlm"
    brain_mode = "vlm" if prefer_vlm and get_vlm_runtime().status()["available"] else "neural" if brain.available else "rules"
    brain.set_mode(brain_mode)
    if brain.available:
        policy, policy_name = default_motor(), "baseline/curious-v2"
    mode_alias = None
    follow_beacon = False
    learned_motor = True
    try:
        await socket.send_json({"type": "scene", "geoms": env.scene()})
        while True:
            tick = asyncio.get_running_loop().time()
            try:
                raw = await asyncio.wait_for(socket.receive_json(), timeout=0.002)
                control = Control.model_validate(raw)
                if control.brain is not None:
                    if control.brain == "neural" and not brain.available:
                        raise ValueError("Bundled planner is unavailable; run setup first")
                    brain.set_mode(control.brain)
                    brain_mode = control.brain
                    mode = "stop"
                    env.grip(False)
                if control.gait is not None:
                    env.motion = control.gait
                if control.energy is not None:
                    env.energy = control.energy
                if control.release:
                    env.grip(False)
                    mode = "stop"
                    if brain_mode == "vlm":
                        brain.cancel_vlm()
                if control.mode is not None:
                    if control.mode in ("manual", "stop", "recover") and brain_mode == "vlm":
                        brain.cancel_vlm()
                    mode = control.mode
                    waypoints = []
                    mission, mission_complete = None, False
                    follow_beacon = control.mode == "follow"
                    mode_alias = control.mode
                    if brain_mode in ("neural", "vlm") and control.mode in ("approach", "follow", "push"):
                        brain.begin(
                            {
                                "approach": "follow the beacon",
                                "follow": "follow me",
                                "push": "push the crate",
                            }[control.mode]
                        )
                        mode = "planner"
                if control.target is not None:
                    target = np.clip(control.target, -5, 5)
                    env.data.mocap_pos[0, :2] = target
                    if mode == "planner":
                        brain.executor.last_plan = -10.0
                    waypoints = []
                    mission, mission_complete = None, False
                if control.text is not None:
                    if not control.text.strip():
                        raise ValueError("Type a command first")
                    if brain_mode in ("neural", "vlm"):
                        brain.begin(control.text)
                        mode, mode_alias, follow_beacon = "planner", None, False
                        env.ball_touched = False
                        try:
                            command = interpret(
                                control.text, env.data.xpos[env.ball], env.data.qpos[:3]
                            )
                            reply = command.reply
                        except ValueError:
                            reply = "Reading the camera and planning your command. Completion is checked in physics."
                        await socket.send_json({"type": "reply", "message": reply})
                    else:
                        command = interpret(
                            control.text,
                            env.data.xpos[env.ball] if env.ball >= 0 else None,
                            env.data.qpos[:3],
                        )
                        mode, waypoints = command.mode, command.waypoints
                        mission = command.goal
                        mission_complete = False
                        if mission == "ball":
                            env.ball_touched = False
                        if waypoints:
                            target = np.asarray(waypoints.pop(0))
                        await socket.send_json({"type": "reply", "message": command.reply})
                manual = (control.forward, control.turn)
                last_control = tick
                if control.paused is not None:
                    paused = control.paused
                if control.reset:
                    obs, _ = env.reset(seed=7)
                    mode, manual = "stop", (0.0, 0.0)
                    waypoints = []
                    mission, mission_complete = None, False
                    brain.close()
                    brain = BrainRouter(env)
                    brain.set_mode(brain_mode)
                    mode_alias, follow_beacon = None, False
                    target = np.array([2.0, 1.0])
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
                        await socket.send_json(
                            {"type": "error", "message": "Unknown trusted policy"}
                        )
            except asyncio.TimeoutError:
                pass
            except ValidationError:
                await socket.send_json({"type": "error", "message": "Invalid controls"})
            except ValueError as error:
                await socket.send_json({"type": "error", "message": str(error)[:200]})
            except TypeError:
                await socket.send_json({"type": "error", "message": "Invalid controls"})
            if mode == "manual" and tick - last_control > 0.5:
                manual = (0.0, 0.0)
            if mode == "follow" or (
                mode == "planner"
                and (follow_beacon or (mode_alias is None and brain.executor.intent == "follow"))
            ):
                target = follow_target(env.data.time)
                env.data.mocap_pos[0, :2] = target
            if (
                mode == "approach"
                and waypoints
                and np.linalg.norm(target - env.data.qpos[:2]) < 0.25
            ):
                target = np.asarray(waypoints.pop(0))
            if mission == "ball" and env.ball_touched and not mission_complete:
                mission_complete = True
                mode = "stop"
                await socket.send_json(
                    {"type": "reply", "message": "Ball booped! Contact detected in MuJoCo."}
                )
            if (
                mission == "behind_ball"
                and not waypoints
                and not mission_complete
                and env.ball >= 0
            ):
                ball = env.data.xpos[env.ball]
                if env.data.qpos[0] > ball[0] + 0.35 and abs(env.data.qpos[1] - ball[1]) < 0.45:
                    mission_complete = True
                    mode = "stop"
                    await socket.send_json(
                        {"type": "reply", "message": "Made it behind the ball (+X side)."}
                    )
            rotation = env.data.xmat[env.torso].reshape(3, 3)
            learned_motor = True
            planner_state = None
            if mode == "planner" and not paused:
                from robopup.decisions import Decision

                try:
                    command, learned_motor = brain.tick()
                    planner_state = brain.status()
                    intent = planner_state["intent"]
                    behavior = mode_alias or {"behind": "approach"}.get(intent, intent)
                    decision = Decision(behavior, command, planner_state["reason"])
                    mission_complete = bool(planner_state["complete"])
                    if brain_mode == "vlm":
                        mission = brain.vlm.instruction
                    elif brain.executor.plan is not None:
                        mission = brain.executor.intent
                except (OSError, RuntimeError, ValueError) as error:
                    decision = Decision(
                        "stop", np.zeros(2), "Planner unavailable: " + str(error)[:120]
                    )
            elif mode == "planner":
                from robopup.decisions import Decision

                planner_state = brain.status()
                decision = Decision(
                    mode_alias or brain.executor.intent, np.zeros(2), brain.executor.reason
                )
            else:
                env.turn_gain = 0.18
                learned_motor = env.motion == "trot"
                decision = decide(
                    mode,
                    env.data.qpos[:3],
                    math.atan2(rotation[1, 0], rotation[0, 0]),
                    env.upright,
                    target,
                    env.data.xpos[env.crate],
                    manual,
                )
            env.command[:] = decision.command
            fallen = env.data.qpos[2] < 0.19 or env.upright < 0.35
            if not paused and not fallen:
                for _ in range(2):
                    # Recompute observations after a command changes.
                    obs = env._obs()
                    action = (
                        np.zeros(12)
                        if policy is None or not learned_motor or np.linalg.norm(env.command) < 0.03
                        else policy.predict(obs, deterministic=True)[0]
                    )
                    obs, _, done, _, _ = env.step(action)
                    if done:
                        break
            if mode == "planner":
                planner_state = brain.status()
            state = env.snapshot()
            state.update(
                {
                    "type": "state",
                    "behavior": decision.behavior,
                    "reason": decision.reason,
                    "target": target.tolist(),
                    "paused": paused,
                    "policy": policy_name,
                    "needs_reset": env.data.qpos[2] < 0.19 or env.upright < 0.35,
                }
            )
            state.update(
                {
                    "mission": mission,
                    "mission_complete": mission_complete,
                    "brain": brain_mode,
                    "planner": planner_state,
                    "motor_source": "PPO residual + analytic trot"
                    if learned_motor and policy is not None
                    else "analytic physical skill",
                }
            )
            await socket.send_json(state)
            await asyncio.sleep(max(0.0, 0.04 - (asyncio.get_running_loop().time() - tick)))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        brain.close()
        env.close()
        active_sessions -= 1


if (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/")
def index():
    if (DIST / "index.html").exists():
        return FileResponse(DIST / "index.html")
    return {"message": "Build the browser first: cd web && npm ci && npm run build"}
