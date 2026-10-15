"""Optional backend math; full compiled stepping is exercised by --probe-only."""
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")

from robopup.mjx_env import gait
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
