"""MuJoCo residual-PPO environment; no rendering or display server required."""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from robopup.model import arena_xml


class RoboPupEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 50}
    dt = .02

    def __init__(self, *, obstacles=False, target_speed=.35, episode_seconds=12,
                 energy_weight=.002, randomize=True):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(arena_xml(obstacles))
        self.data = mujoco.MjData(self.model)
        self.target_speed = float(target_speed)
        self.energy_weight = float(energy_weight)
        self.max_steps = round(episode_seconds / self.dt)
        self.randomize = randomize
        self.action_space = spaces.Box(-1., 1., (12,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (50,), dtype=np.float32)
        self.command = np.array([self.target_speed, 0.], dtype=np.float64)
        self.previous_action = np.zeros(12)
        self.steps = 0
        self.phase = 0.
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.crate = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "crate")

    @property
    def upright(self):
        return float(self.data.xmat[self.torso].reshape(3, 3)[2, 2])

    def gait(self):
        """Analytic diagonal trot + yaw offset. PPO supplies joint residuals."""
        speed, turn = self.command
        out = []
        moving = abs(speed) + abs(turn) > .03
        for i, offset in enumerate((0., np.pi, np.pi, 0.)):
            p = self.phase + offset
            side = 1 if i in (0, 2) else -1
            stride = np.clip(speed - side * turn * .18, -.6, .6) * .22
            x = stride * np.sin(p) if moving else 0.
            z = -.365 + (.055 * max(0., np.cos(p)) if moving else 0.)
            r = np.hypot(x, z)
            knee = -np.arccos(np.clip((r*r - .2**2 - .21**2) / (2*.2*.21), -1, 1))
            hip = np.arctan2(-x, -z) - np.arctan2(.21*np.sin(knee), .2+.21*np.cos(knee))
            out.extend((0., hip, knee))
        return np.asarray(out)

    def _obs(self):
        rotation = self.data.xmat[self.torso].reshape(3, 3)
        return np.concatenate((
            self.data.qvel[:6], rotation[2], self.data.qpos[7:19],
            self.data.qvel[6:18], self.command,
            [np.sin(self.phase), np.cos(self.phase), self.data.qpos[2]],
            self.previous_action,
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps, self.phase = 0, 0.
        self.command[:] = (self.target_speed, 0.)
        self.previous_action[:] = 0
        self.data.qpos[7:19] = self.gait()
        if self.randomize:
            self.data.qpos[7:19] += self.np_random.uniform(-.025, .025, 12)
            self.model.geom_friction[0, 0] = self.np_random.uniform(.7, 1.1)
        self.data.ctrl[:] = self.data.qpos[7:19]
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Expected 12 finite joint residuals")
        action = np.clip(action, -1, 1)
        self.phase = (self.phase + self.dt * 2*np.pi*2.) % (2*np.pi)
        targets = self.gait() + action * .3
        self.data.ctrl[:] = np.clip(targets, self.model.jnt_range[1:13, 0],
                                   self.model.jnt_range[1:13, 1])
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        rotation = self.data.xmat[self.torso].reshape(3, 3)
        local_v = rotation.T @ self.data.qvel[:3]
        velocity_error = (local_v[0] - self.command[0])**2
        track = float(np.exp(-velocity_error / .08))
        energy = float(np.mean(np.square(self.data.actuator_force)))
        smooth = float(np.mean(np.square(action - self.previous_action)))
        terminated = bool(self.data.qpos[2] < .19 or self.upright < .35
                          or not np.isfinite(self.data.qpos).all())
        reward = 1.5 * track + .5 * self.upright - self.energy_weight*energy - .05*smooth
        if terminated:
            reward -= 2.
        self.previous_action[:] = action
        return self._obs(), float(reward), terminated, self.steps >= self.max_steps, {
            "speed": float(local_v[0]), "upright": self.upright,
            "tracking": track, "energy": energy,
            "distance": float(self.data.qpos[0]), "fallen": terminated,
        }

    def snapshot(self):
        return {
            "time": float(self.data.time), "position": self.data.qpos[:3].tolist(),
            "upright": self.upright,
            "speed": float(np.linalg.norm(self.data.qvel[:2])),
            "crate": self.data.xpos[self.crate].tolist(),
            "geoms": [{"id": i, "position": self.data.geom_xpos[i].tolist(),
                       "rotation": self.data.geom_xmat[i].tolist()}
                      for i in range(1, self.model.ngeom)],
        }

    def scene(self):
        names = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
                 int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
                 int(mujoco.mjtGeom.mjGEOM_BOX): "box",
                 int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder"}
        return [{"id": i, "name": mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i),
                 "type": names.get(int(self.model.geom_type[i]), "box"),
                 "size": self.model.geom_size[i].tolist(),
                 "color": self.model.geom_rgba[i].tolist()}
                for i in range(1, self.model.ngeom)]
