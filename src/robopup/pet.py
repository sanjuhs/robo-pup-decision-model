"""Physical pet skills: motor-driven gaits/jump and contact-gated weld gripper.

The gripper is a simulated magnetic tool, not an anatomical mouth. A grasp adds
a MuJoCo equality constraint at the contact pose; it never teleports the ball.
"""

import mujoco
import numpy as np

from robopup.env import RoboPupEnv


class PetEnv(RoboPupEnv):
    def __init__(self, **kwargs):
        self.motion = "trot"
        self.jump_time = None
        super().__init__(obstacles=True, pet=True, randomize=False, **kwargs)
        self.gripper = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self.gripper_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_geom")
        self.lift_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "grip_lift")
        self.weld = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "ball_grasp")
        self.feet = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n + "_foot")
            for n in ("fl", "fr", "rl", "rr")
        }
        self.energy = 1.0
        self.magnet = False
        self.grasp_count = 0
        self.jump_max_height = 0.0
        self.jump_flight_seconds = 0.0
        self.turn_gain = 0.18
        self.walk_lift = 0.07
        self.walk_frequency = 1.5
        self.jump_gain = 90.0
        self.jump_force = 28.0
        self.gait_air_seconds = {"walk": 0.0, "trot": 0.0, "run": 0.0}
        self.lowering = False
        self.motion_blend = 0.0
        self.previous_motion = "trot"

    @property
    def held(self):
        return bool(self.data.eq_active[self.weld])

    @property
    def foot_contacts(self):
        return {g for c in self.data.contact for g in (c.geom1, c.geom2) if g in self.feet}

    def reset(self, **kwargs):
        self.motion, self.jump_time = "trot", None
        obs, info = super().reset(**kwargs)
        self.data.qpos[self.model.jnt_qposadr[self.lift_joint]] = 0.2
        self.data.ctrl[12] = 0.2
        self.data.eq_active[self.weld] = 0
        self.energy, self.magnet, self.grasp_count = 1.0, False, 0
        self.jump_max_height, self.jump_flight_seconds = 0.0, 0.0
        self.gait_air_seconds = {"walk": 0.0, "trot": 0.0, "run": 0.0}
        self.lowering = False
        self.motion_blend = 0.0
        self.previous_motion = "trot"
        mujoco.mj_forward(self.model, self.data)
        return obs, info

    def grip(self, enabled):
        self.magnet = bool(enabled)
        if not enabled:
            self.data.eq_active[self.weld] = 0

    def _grasp_at_contact(self):
        if self.held or not self.magnet:
            return
        if not any(
            {c.geom1, c.geom2} == {self.ball_geom, self.gripper_geom} for c in self.data.contact
        ):
            return
        a, b = self.gripper, self.ball
        rotation = self.data.xmat[a].reshape(3, 3)
        self.model.eq_data[self.weld, 3:6] = rotation.T @ (self.data.xpos[b] - self.data.xpos[a])
        inverse, relative = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(inverse, self.data.xquat[a])
        mujoco.mju_mulQuat(relative, inverse, self.data.xquat[b])
        self.model.eq_data[self.weld, 6:10] = relative
        self.data.eq_active[self.weld] = 1
        self.grasp_count += 1

    def start_jump(self):
        if (
            self.jump_time is not None
            or self.held
            or self.energy < 0.2
            or len(self.foot_contacts) < 2
        ):
            return False
        self.jump_time = 0.0
        self.jump_max_height = float(self.data.qpos[2])
        self.jump_flight_seconds = 0.0
        return True

    @staticmethod
    def _ik(x, z):
        r = np.hypot(x, z)
        knee = -np.arccos(np.clip((r * r - 0.2**2 - 0.21**2) / (2 * 0.2 * 0.21), -1, 1))
        hip = np.arctan2(-x, -z) - np.arctan2(0.21 * np.sin(knee), 0.2 + 0.21 * np.cos(knee))
        return (0.0, hip, knee)

    def gait(self):
        if self.jump_time is not None:
            t = self.jump_time
            if t < 0.3:
                z = -0.365 + 0.12 * t / 0.3
            elif t < 0.38:
                z = -0.245
            elif t < 0.50:
                z = -0.245 - 0.16 * (t - 0.38) / 0.12
            elif t < 0.85:
                z = -0.34
            else:
                z = -0.365
            return np.tile(self._ik(0.0, z), 4)
        if self.motion == "trot" and self.turn_gain == 0.18:
            return super().gait()
        moving = np.linalg.norm(self.command) > 0.03
        offsets = (
            (0.0, np.pi, np.pi / 2, 3 * np.pi / 2)
            if self.motion == "walk"
            else (0.0, np.pi, np.pi, 0.0)
        )
        result = []
        for i, offset in enumerate(offsets):
            p = (self.phase + offset) % (2 * np.pi)
            side = 1 if i in (0, 2) else -1
            stride = np.clip(
                self.command[0] - side * self.command[1] * self.turn_gain, -0.6, 0.6
            ) * (0.32 if self.motion == "walk" else 0.22)
            if self.motion == "walk":
                x = stride * np.sin(p)
                lift = self.walk_lift * max(0.0, np.cos(p))
            else:
                x, lift = (
                    stride * np.sin(p),
                    (0.075 if self.motion == "run" else 0.055) * max(0.0, np.cos(p)),
                )
            blend = self.motion_blend
            result.extend(
                self._ik(x * blend if moving else 0.0, -0.365 + (lift * blend if moving else 0.0))
            )
        return np.array(result)

    def step(self, action):
        self.data.ctrl[12] = (
            0.0 if self.lowering else (0.2 if self.held or not self.magnet else 0.0)
        )
        if self.previous_motion != self.motion or np.linalg.norm(self.command) < 0.03:
            self.motion_blend = 0.0
            self.previous_motion = self.motion
        self.motion_blend = min(1.0, self.motion_blend + self.dt / 0.5)
        if self.jump_time is not None:
            self.command[:] = 0.0
            action = np.zeros(12)
        gain = self.jump_gain if self.jump_time is not None else 45.0
        force = self.jump_force if self.jump_time is not None else 18.0
        self.model.actuator_gainprm[:12, 0] = gain
        self.model.actuator_biasprm[:12, 1] = -gain
        self.model.actuator_forcerange[:12, 0] = -force
        self.model.actuator_forcerange[:12, 1] = force
        self.gait_frequency = {"walk": self.walk_frequency, "trot": 2.0, "run": 3.2}[self.motion]
        result = super().step(action)
        self._grasp_at_contact()
        if (
            self.jump_time is None
            and np.linalg.norm(self.command) > 0.03
            and not self.foot_contacts
        ):
            self.gait_air_seconds[self.motion] += self.dt
        effort = float(np.mean(np.abs(self.data.actuator_force[:12] * self.data.qvel[6:18])))
        resting = np.linalg.norm(self.command) < 0.03 and self.jump_time is None
        self.energy = float(
            np.clip(self.energy + self.dt * (0.012 if resting else -0.002 - 0.0003 * effort), 0, 1)
        )
        if self.jump_time is not None:
            self.jump_time += self.dt
            self.jump_max_height = max(self.jump_max_height, float(self.data.qpos[2]))
            if not self.foot_contacts:
                self.jump_flight_seconds += self.dt
            if self.jump_time >= 1.6:
                self.jump_time = None
        return result

    def snapshot(self):
        state = super().snapshot()
        state.update(
            {
                "energy": self.energy,
                "held": self.held,
                "gait": self.motion,
                "jump_active": self.jump_time is not None,
                "jump_max_height": self.jump_max_height,
                "jump_flight_seconds": self.jump_flight_seconds,
                "gripper": "contact-gated magnetic weld",
                "foot_contacts": [
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g).split("_")[0]
                    for g in sorted(self.foot_contacts)
                ],
                "gait_air_seconds": self.gait_air_seconds.copy(),
            }
        )
        return state
