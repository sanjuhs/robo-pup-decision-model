import mujoco
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from robopup.env import RoboPupEnv
from robopup.experiments import participant_path


def test_environment_contract_and_seed():
    env = RoboPupEnv()
    check_env(env)
    a, _ = env.reset(seed=13)
    b, _ = env.reset(seed=13)
    np.testing.assert_array_equal(a, b)
    assert env.model.nu == 12
    assert env.observation_space.contains(a)
    scene = env.scene()
    assert next(g for g in scene if g["name"] == "head")["body"] == "torso"
    assert next(g for g in scene if g["name"] == "fl_foot")["body"] == "fl_shin"


def test_gait_rollout_and_contacts():
    env = RoboPupEnv(randomize=False)
    env.reset(seed=1)
    for _ in range(250):
        obs, reward, done, _, _ = env.step(np.zeros(12))
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if done:
            break
    assert env.data.time > .5
    assert env.data.ncon > 0
    assert env.snapshot()["geoms"]


def test_floor_crate_contact_is_physical():
    env = RoboPupEnv()
    env.reset(seed=1)
    for _ in range(100):
        mujoco.mj_step(env.model, env.data)
    assert .12 < env.data.xpos[env.crate, 2] < .16


def test_fall_terminates_and_invalid_action_rejected():
    env = RoboPupEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(np.full(12, np.nan))
    env.data.qpos[2] = .1
    mujoco.mj_forward(env.model, env.data)
    assert env.step(np.zeros(12))[2]


def test_participant_paths_cannot_escape(tmp_path):
    assert participant_path(tmp_path, "team-1", "v1").parent == tmp_path / "team-1"
    with pytest.raises(ValueError):
        participant_path(tmp_path, "../outside", "run")
