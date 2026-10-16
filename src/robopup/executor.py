"""Closed-loop skill execution. Learned image targets; contact/pose goal verification.

This module deliberately separates the neural proposal, analytic skill mechanics,
and physical completion checks. It never reads true object positions to navigate.
"""

import math

import numpy as np


def heading(env):
    r = env.data.xmat[env.torso].reshape(3, 3)
    return math.atan2(r[1, 0], r[0, 0])


class GoalExecutor:
    def __init__(self, env):
        self.env = env
        self.instruction = "stop"
        self.intent = "stop"
        self.plan = None
        self.target = None
        self.phase = "idle"
        self.complete = False
        self.failed = False
        self.reason = "Ready for a command"
        self.started = 0.0
        self.last_plan = -10.0
        self.origin = np.zeros(2)
        self.origin_yaw = 0.0
        self.release_time = 0.0
        self.ever_held = False
        self.maximum_ball_height = 0.0
        self.grasp_position = None
        self.last_image = None
        self.resting = False
        self.route_side = None
        self.behind_final = False
        self.settle_time = 0.0

    def begin(self, instruction):
        self.instruction = instruction[:160]
        self.plan, self.target = None, None
        self.complete = self.failed = self.ever_held = self.resting = False
        self.maximum_ball_height = 0.0
        self.grasp_position = None
        self.phase = "planning"
        self.started = self.env.data.time
        self.last_plan = -10.0
        self.origin = self.env.data.qpos[:2].copy()
        self.origin_yaw = heading(self.env)
        self.reason = "Reading camera and instruction"
        self.env.ball_touched = False
        self.route_side = None
        self.behind_final = False
        self.env.lowering = False

    def update_plan(self, plan, image=None):
        self.plan = plan
        self.last_image = image
        self.last_plan = self.env.data.time
        self.intent = plan["intent"]
        row = plan["rows"][0]
        self.target = self.env.data.qpos[:2] + np.asarray(row["delta"])
        if row["skill"] == "rest" and self.intent != "rest":
            self.resting = True
        self.reason = f"RGB transformer: {row['skill']} · {row['object']}"

    def _navigate(self, target, stop_distance=0.12):
        e = self.env
        delta = np.asarray(target) - e.data.qpos[:2]
        distance = float(np.linalg.norm(delta))
        angle = math.atan2(delta[1], delta[0]) - heading(e)
        error = math.atan2(math.sin(angle), math.cos(angle))
        turn = float(np.clip(error * 2.5, -1.2, 1.2))
        if distance < stop_distance:
            return np.zeros(2)
        # The small forward component compensates the measured backwards drift
        # of this differential-stride turn; the pose still comes from contacts.
        forward = 0.15 * abs(turn) if abs(error) > 0.45 else min(0.4, max(0.15, distance * 0.65))
        return np.array([forward, turn])

    def command(self):
        e = self.env
        e.turn_gain = 0.5
        if self.complete or self.failed:
            return np.zeros(2)
        if e.upright < 0.45 or e.data.qpos[2] < 0.19:
            self.failed, self.reason = True, "Fallen. Reset posture to continue."
            return np.zeros(2)
        if e.data.time - self.started > 150:
            self.failed, self.reason = True, "Goal timed out; no completion claimed"
            e.grip(False)
            return np.zeros(2)
        if self.phase == "settling":
            if e.data.time - self.settle_time > 0.7 and np.linalg.norm(e.data.qvel[:6]) < 0.15:
                self.phase = "lowering"
                self.settle_time = e.data.time
                e.lowering = True
            return np.zeros(2)
        if self.phase == "lowering":
            if e.data.time - self.settle_time > 0.8:
                self.phase, self.release_time = "release", e.data.time
                e.grip(False)
            return np.zeros(2)
        if self.phase == "release":
            e.grip(False)
            if e.data.time - self.release_time > 1.2:
                # This is an outcome verifier, not a navigation target.
                ball = e.data.xpos[e.ball]
                self.complete = bool(
                    np.linalg.norm(ball[:2]) < 0.6
                    and ball[2] < 0.19
                    and self.ever_held
                    and self.maximum_ball_height > 0.2
                )
                self.failed = not self.complete
                self.reason = (
                    "Fetch complete: contact, lift, return and release verified"
                    if self.complete
                    else "Released, but return criterion failed"
                )
            return np.zeros(2)
        if self.plan is None:
            return np.zeros(2)
        row = self.plan["rows"][0]
        if self.plan["confidence"] < 0.65 or row["confidence"] < 0.65:
            self.reason = "Uncertain instruction; holding position"
            return np.zeros(2)
        if self.intent == "unsupported":
            self.failed, self.reason = (
                True,
                "Unsupported instruction; try a command from the field guide",
            )
            return np.zeros(2)
        if self.resting:
            if e.energy < 0.45:
                self.reason = "Resting to recharge virtual energy"
                return np.zeros(2)
            self.resting = False
        if self.intent in ("stop", "rest", "drop"):
            if self.intent == "drop":
                e.grip(False)
            self.complete = (
                e.energy > 0.9
                if self.intent == "rest"
                else np.linalg.norm(e.data.qvel[:2]) < 0.08
                if self.intent == "stop"
                else True
            )
            self.reason = (
                "Holding a standing pose"
                if self.intent == "stop"
                else "Resting"
                if self.intent == "rest"
                else "Gripper released"
            )
            return np.zeros(2)
        if self.intent == "jump":
            if self.phase == "planning":
                if e.start_jump():
                    self.phase = "jumping"
            elif e.jump_time is None:
                self.complete = (
                    e.jump_flight_seconds > 0.04 and len(e.foot_contacts) >= 2 and e.upright > 0.9
                )
                self.failed = not self.complete
                self.reason = (
                    "Hop and landing verified from contacts"
                    if self.complete
                    else "Jump did not meet its flight/landing criteria"
                )
            return np.zeros(2)
        if self.intent in ("walk", "run"):
            self.phase = self.intent
            e.motion = self.intent
            goal_distance = 0.5 if self.intent == "walk" else 1.2
            direction = np.array([math.cos(self.origin_yaw), math.sin(self.origin_yaw)])
            if np.dot(e.data.qpos[:2] - self.origin, direction) >= goal_distance:
                self.complete, self.reason = True, f"{self.intent.title()} distance reached"
                return np.zeros(2)
            turn_error = self.origin_yaw - heading(e)
            return np.array(
                [
                    0.35 if self.intent == "walk" else 0.55,
                    np.clip(math.atan2(math.sin(turn_error), math.cos(turn_error)) * 2, -1, 1),
                ]
            )
        e.motion = "trot"
        if not row["visible"]:
            self.reason = f"{row['object']} not visible in the camera; holding position"
            return np.zeros(2)
        if self.intent == "fetch":
            e.grip(True)
            if e.held:
                self.ever_held = True
                self.maximum_ball_height = max(
                    self.maximum_ball_height, float(e.data.xpos[e.ball, 2])
                )
                if self.grasp_position is None:
                    self.grasp_position = e.data.qpos[:2].copy()
                self.phase = "returning"
                if np.linalg.norm(e.data.qpos[:2]) < 0.28 and self.maximum_ball_height > 0.2:
                    self.phase, self.settle_time = "settling", e.data.time
                    return np.zeros(2)
                # Wait for the next image-based home plan; don't navigate using a stale ball row.
                if row["object"] != "home":
                    return np.zeros(2)
                return self._navigate(self.target, 0.12)
            self.phase = "retrieving"
            return self._navigate(self.target, 0.38)
        if self.intent == "approach" and e.ball_touched:
            self.complete, self.reason = True, "Ball contact verified in MuJoCo"
            return np.zeros(2)
        if self.intent == "behind" and np.linalg.norm(self.target - e.data.qpos[:2]) < 0.18:
            ball = e.data.xpos[e.ball]
            self.complete = bool(
                e.data.qpos[0] > ball[0] + 0.35 and abs(e.data.qpos[1] - ball[1]) < 0.45
            )
            if self.complete:
                self.reason = "Behind-ball position verified (+X side)"
                return np.zeros(2)
        if self.intent == "behind":
            estimated_ball = (
                self.target
                - np.asarray(row["delta"])
                + np.asarray(row.get("object_delta", row["delta"]))
            )
            if self.route_side is None:
                self.route_side = 1.0 if e.data.qpos[1] >= estimated_ball[1] else -1.0
            if e.data.qpos[0] > estimated_ball[0] + 0.45:
                self.behind_final = True
            if not self.behind_final:
                waypoint = estimated_ball + np.array([0.65, self.route_side * 0.8])
                if (
                    e.data.qpos[0] < estimated_ball[0] - 0.35
                    and abs(e.data.qpos[1] - estimated_ball[1]) < 0.55
                ):
                    waypoint = estimated_ball + np.array([-0.45, self.route_side * 0.8])
                return self._navigate(waypoint, 0.1)
        if self.intent == "push":
            # Goal is part of the task specification; crate estimate is from RGB.
            crate = self.target
            goal = np.array([3.0, 0.0])
            direction = goal - crate
            distance = np.linalg.norm(direction)
            if distance < 0.3:
                self.complete = bool(np.linalg.norm(e.data.xpos[e.crate, :2] - goal) < 0.35)
                if self.complete:
                    self.reason = "Crate-to-ring position verified"
                    return np.zeros(2)
            direction /= max(distance, 1e-6)
            behind = crate - 0.65 * direction
            aligned = (
                np.dot(crate - e.data.qpos[:2], direction) > 0.15
                and abs(np.cross(direction, crate - e.data.qpos[:2])) < 0.18
            )
            return self._navigate(crate + 0.25 * direction if aligned else behind, 0.1)
        if self.intent == "follow":
            self.phase = "tracking"
            return self._navigate(self.target, 0.4)
        return self._navigate(self.target, 0.1)

    def observe(self):
        """Synchronize outcome telemetry after the latest physical step."""
        if self.intent == "fetch" and self.env.held:
            self.ever_held = True
            self.maximum_ball_height = max(
                self.maximum_ball_height, float(self.env.data.xpos[self.env.ball, 2])
            )

    def status(self):
        self.observe()
        return {
            "instruction": self.instruction,
            "intent": self.intent,
            "phase": self.phase,
            "complete": bool(self.complete),
            "failed": bool(self.failed),
            "reason": self.reason,
            "plan": self.plan,
            "elapsed_s": self.env.data.time - self.started,
            "ever_held": self.ever_held,
            "maximum_ball_height": self.maximum_ball_height,
        }
