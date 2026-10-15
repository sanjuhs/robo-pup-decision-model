"""Inspectable behavior selector. No neural decision model is implied."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class Decision:
    behavior: str
    command: np.ndarray
    reason: str


def follow_target(seconds):
    """Slow beacon, shared by the arena and evaluation protocol (0.072 m/s)."""
    return np.array([1.2*math.cos(seconds*.06), 1.2*math.sin(seconds*.06)])


def decide(mode, position, yaw, upright, target, crate, manual=(0., 0.)):
    if upright < .45 or position[2] < .19:
        return Decision("recover", np.zeros(2), "Fallen: reset posture to continue")
    if mode == "manual":
        return Decision("manual", np.clip(manual, [-.4, -1.5], [.6, 1.5]), "Keyboard control")
    if mode in ("stop", "recover"):
        return Decision(mode, np.zeros(2), "Hold standing pose" if mode == "stop" else "Hold posture; reset if fallen")
    waypoint = np.asarray(target[:2], dtype=float)
    reason = "Walk toward the treat" if mode == "approach" else "Track the moving beacon"
    if mode == "push":
        goal = np.array([3., 0.])
        direction = goal - np.asarray(crate[:2])
        if np.linalg.norm(direction) < .25:
            return Decision("stop", np.zeros(2), "Crate reached the goal")
        direction /= max(np.linalg.norm(direction), 1e-6)
        behind = np.asarray(crate[:2]) - .55*direction
        waypoint = behind if np.linalg.norm(behind-position[:2]) > .25 else np.asarray(crate[:2])+.3*direction
        reason = "Line up behind the crate, then make contact (experimental)"
    delta = waypoint - position[:2]
    distance = np.linalg.norm(delta)
    error = math.atan2(math.sin(math.atan2(delta[1], delta[0])-yaw),
                       math.cos(math.atan2(delta[1], delta[0])-yaw))
    if distance < .2:
        return Decision("stop", np.zeros(2), "Treat reached")
    speed = min(.35, max(.18, distance*.6)) * max(0., math.cos(error))
    return Decision(mode, np.array([speed, np.clip(error*1.5, -1.2, 1.2)]), reason)
