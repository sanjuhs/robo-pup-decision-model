"""Optional batched MJX physics with the native environment's policy interface.

Only imported by the explicit MJX training path. JAX selects CPU/GPU at runtime;
--require-gpu in the runner refuses an accidental CPU-only paid experiment.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from robopup.env import RoboPupEnv


def gait(phase, command):
    offsets = jp.array([0., jp.pi, jp.pi, 0.])
    side = jp.array([1., -1., 1., -1.])
    p = phase + offsets
    moving = jp.abs(command[0])+jp.abs(command[1]) > .03
    stride = jp.clip(command[0]-side*command[1]*.18, -.6, .6)*.22
    x = jp.where(moving, stride*jp.sin(p), 0.)
    z = -.365 + jp.where(moving, .055*jp.maximum(0., jp.cos(p)), 0.)
    r2 = x*x+z*z
    knee = -jp.arccos(jp.clip((r2-.2**2-.21**2)/(2*.2*.21), -1., 1.))
    hip = jp.arctan2(-x, -z)-jp.arctan2(.21*jp.sin(knee), .2+.21*jp.cos(knee))
    return jp.stack((jp.zeros(4), hip, knee), axis=1).reshape(12)


def observation(data, phase, action, command, torso):
    return jp.concatenate((data.qvel[:6], data.xmat[torso, 2], data.qpos[7:19],
                           data.qvel[6:18], command,
                           jp.array([jp.sin(phase), jp.cos(phase), data.qpos[2]]), action))


class MJXVecEnv(VecEnv):
    """SB3 VecEnv adapter; physics batched on the JAX device, PPO kept separate."""
    render_mode = None

    def __init__(self, n_envs=64, seed=7, target_speed=.35, energy_weight=.002):
        native = RoboPupEnv(target_speed=target_speed, energy_weight=energy_weight)
        self.native_model = native.model
        # CG is supported by both engines and limits GPU solve work per frame.
        self.native_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self.native_model.opt.iterations = 10
        self.native_model.opt.ls_iterations = 4
        self.mjx_model = mjx.put_model(self.native_model)
        self.torso = native.torso
        self.command = jp.array([target_speed, 0.], dtype=jp.float32)
        self.energy_weight = energy_weight
        self.seed_value = seed
        self.max_steps = native.max_steps
        self.initial = mjx.make_data(self.mjx_model)
        self._reset_batch = jax.jit(jax.vmap(self._reset_one))
        self._step_batch = jax.jit(jax.vmap(self._step_one))
        super().__init__(n_envs, native.observation_space, native.action_space)
        self.actions = np.zeros((n_envs, 12), dtype=np.float32)
        native.close()

    def _reset_one(self, key):
        qpos = self.initial.qpos.at[7:19].set(gait(jp.float32(0), self.command)
                    + jax.random.uniform(key, (12,), minval=-.025, maxval=.025))
        data = mjx.forward(self.mjx_model, self.initial.replace(qpos=qpos, ctrl=qpos[7:19]))
        return data, jp.float32(0), jp.zeros(12), jp.int32(0), key

    def _step_one(self, state, action):
        data, phase, previous, steps, key = state
        action = jp.clip(action, -1., 1.)
        phase = (phase+.02*2*jp.pi*2.) % (2*jp.pi)
        targets = jp.clip(gait(phase, self.command)+action*.3,
                          self.mjx_model.jnt_range[1:13, 0], self.mjx_model.jnt_range[1:13, 1])
        data = data.replace(ctrl=targets)
        data = jax.lax.fori_loop(0, 4, lambda _, d: mjx.step(self.mjx_model, d), data)
        steps += 1
        local_v = data.xmat[self.torso].T @ data.qvel[:3]
        upright = data.xmat[self.torso, 2, 2]
        tracking = jp.exp(-(local_v[0]-self.command[0])**2/.08)
        energy = jp.mean(data.actuator_force**2)
        smooth = jp.mean((action-previous)**2)
        fallen = (data.qpos[2]<.19) | (upright<.35) | (~jp.all(jp.isfinite(data.qpos)))
        truncated = steps >= self.max_steps
        done = fallen | truncated
        reward = 1.5*tracking+.5*upright-self.energy_weight*energy-.05*smooth-2.*fallen
        obs = observation(data, phase, action, self.command, self.torso)
        new_state = data, phase, action, steps, key
        key, reset_key = jax.random.split(key)
        # Conditional reset is evaluated per world; no host-side physics loop.
        new_state = jax.lax.cond(done, lambda: self._reset_one(reset_key),
                                 lambda: (*new_state[:4], key))
        reset_obs = observation(new_state[0], new_state[1], new_state[2], self.command, self.torso)
        return new_state, reset_obs, reward, done, truncated & (~fallen), obs

    def reset(self):
        self.state = self._reset_batch(jax.random.split(jax.random.PRNGKey(self.seed_value), self.num_envs))
        obs = jax.vmap(partial(observation, command=self.command, torso=self.torso))(
            self.state[0], self.state[1], self.state[2])
        return np.asarray(obs, dtype=np.float32)

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.state, obs, rewards, dones, trunc, terminal = self._step_batch(
            self.state, jp.asarray(self.actions))
        obs, rewards, dones, trunc, terminal = jax.device_get((obs, rewards, dones, trunc, terminal))
        infos = [{"TimeLimit.truncated": bool(trunc[i]),
                  **({"terminal_observation": terminal[i]} if dones[i] else {})}
                 for i in range(self.num_envs)]
        return np.asarray(obs, dtype=np.float32), np.asarray(rewards), np.asarray(dones), infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None) for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("MJX environment parameters are fixed for each run")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("MJX environment exposes reset and step only")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]
