"""Optional backend math; full compiled stepping is exercised by --probe-only."""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from robopup.mjx_env import gait, MJXVecEnv
from robopup.env import RoboPupEnv


def test_mjx_and_native_gait_targets_agree():
    env = RoboPupEnv()
    env.reset(seed=1)
    for phase in (0., .7, 2.3):
        env.phase = phase
        for command in ([0., 0.], [.35, 0.], [.2, -.7]):
            env.command[:] = command
            np.testing.assert_allclose(np.asarray(gait(phase, np.asarray(command))),
                                       env.gait(), atol=1e-6)


def test_sb3_timeout_bootstrap_can_modify_mjx_host_arrays():
    # Exercise the adapter's public boundary with the immutable arrays JAX returns.
    env = object.__new__(MJXVecEnv)
    env.num_envs = 1
    env.state = None
    env.actions = np.zeros((1,12))
    obs = np.zeros((1,50),dtype=np.float32)
    rewards = np.ones(1,dtype=np.float32)
    dones = np.ones(1,dtype=bool)
    for array in (obs,rewards,dones):
        array.setflags(write=False)
    env._step_batch = lambda state, actions: (None,obs,rewards,dones,dones,obs)
    returned_obs, returned_rewards, returned_done, info = env.step_wait()
    returned_rewards[returned_done] += .99
    assert returned_rewards[0] > 1.9
    assert returned_obs.flags.writeable
    assert info[0]["TimeLimit.truncated"]
