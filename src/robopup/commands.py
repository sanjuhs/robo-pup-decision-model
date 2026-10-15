"""A small local command grammar, not an LLM or a promise of task success."""
from dataclasses import dataclass, field
import re

import numpy as np


@dataclass
class PetCommand:
    mode: str
    reply: str
    waypoints: list[list[float]] = field(default_factory=list)
    goal: str | None = None


def interpret(text: str, ball, position) -> PetCommand:
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    words = set(text.split())
    if not words:
        raise ValueError("Try: go to the ball, go behind the ball, follow me, or stop")
    if words & {"stop", "stay", "wait", "halt"}:
        return PetCommand("stop", "I'll hold my standing pose.")
    if words & {"recover", "up", "reset"}:
        return PetCommand("recover", "Use Reset posture after a tumble; I haven't learned self-righting.")
    if "ball" in words:
        if ball is None:
            raise ValueError("There is no ball in this physics world")
        x, y = float(ball[0]), float(ball[1])
        if "behind" in words:
            # Behind is defined as the +X side. Route around the near side of the ball.
            side = -1. if position[1] <= y else 1.
            points = [[x-.65, y+side*.7], [x+.7, y+side*.7], [x+.7, y]]
            return PetCommand("approach", "I'll try to go around to the far (+X) side of the ball.", points, "behind_ball")
        # Aim for the ball itself; the physics contact sensor ends the mission.
        # A stand-off target can stop before a foot actually reaches the ball.
        point = np.asarray(ball[:2], dtype=float).tolist()
        reply = "I'll investigate the ball. Robots don't eat balls!" if words & {"eat", "sniff"} else "I'll try to reach the ball."
        return PetCommand("approach", reply, [point], "ball")
    if "follow" in words:
        return PetCommand("follow", "I'll try to follow the moving beacon; this skill is experimental.")
    if words & {"push", "nudge"}:
        return PetCommand("push", "I'll try to push the crate toward the orange ring. This is experimental.")
    if words & {"treat", "target"}:
        return PetCommand("approach", "I'll try to find the treat you placed.")
    raise ValueError("Try: go to the ball, go behind the ball, follow me, push the crate, or stop")
