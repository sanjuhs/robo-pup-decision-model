"""Regression gates for the trained checkpoint, not an untrained tensor scaffold."""

from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from robopup.executor import GoalExecutor, heading
from robopup.pet import PetEnv
from robopup.scene_planner import infer, load_planner
from robopup.vision import PlannerCamera

ROOT = Path(__file__).resolve().parents[1]


def test_learned_modalities_change_plan_and_unseen_object_abstains():
    torch.set_num_threads(1)
    model, _ = load_planner(ROOT / "demo/arena-planner")
    env = PetEnv()
    env.reset(seed=42)
    env.command[:] = 0
    for _ in range(25):
        env.step(np.zeros(12))
    camera = PlannerCamera(env)
    try:
        image = camera.capture()

        def plan(text="fetch the ball", energy=1, held=0, rgb=None):
            return infer(model, image if rgb is None else rgb, text, env._obs(), energy, held)

        fetch = plan()
        assert fetch["intent"] == "fetch" and fetch["rows"][0]["skill"] == "retrieve"
        assert plan("push the crate")["rows"][0]["object"] == "crate"
        assert plan(energy=0.05)["rows"][0]["skill"] == "rest"
        assert plan(held=1)["rows"][0]["skill"] == "return"
        shifted = plan(rgb=np.roll(image, 8, axis=1))
        assert (
            abs(shifted["rows"][0]["object_delta"][0] - fetch["rows"][0]["object_delta"][0] - 0.5)
            < 0.02
        )
        blank = plan(rgb=np.zeros_like(image))
        assert not blank["rows"][0]["visible"]
        ex = GoalExecutor(env)
        ex.begin("fetch the ball")
        ex.update_plan(blank)
        np.testing.assert_array_equal(ex.command(), np.zeros(2))
    finally:
        camera.close()
        env.close()


def test_rgb_transformer_physically_fetches_with_contact_lift_and_release():
    torch.set_num_threads(1)
    model, _ = load_planner(ROOT / "demo/arena-planner")
    motor = PPO.load(ROOT / "demo/curious-v2/policy.zip", device="cpu")
    env = PetEnv(episode_seconds=125)
    env.reset(seed=7)
    env.command[:] = 0
    for _ in range(25):
        env.step(np.zeros(12))
    ex = GoalExecutor(env)
    ex.begin("fetch the ball")
    camera = PlannerCamera(env)
    try:
        while env.data.time < 120 and not ex.complete and not ex.failed:
            if env.data.time - ex.last_plan >= 0.5:
                ex.update_plan(
                    infer(
                        model,
                        camera.capture(),
                        ex.instruction,
                        env._obs(),
                        env.energy,
                        float(env.held),
                        heading(env),
                    )
                )
            env.command[:] = ex.command()
            use_ppo = (
                env.motion == "trot"
                and abs(env.command[1]) < 0.25
                and np.linalg.norm(env.command) > 0.03
            )
            env.step(motor.predict(env._obs(), deterministic=True)[0] if use_ppo else np.zeros(12))
        assert ex.complete, ex.reason
        assert ex.ever_held and env.grasp_count == 1 and ex.maximum_ball_height > 0.2
        assert not env.held and np.linalg.norm(env.data.xpos[env.ball, :2]) < 0.6
        assert env.data.xpos[env.ball, 2] < 0.19 and env.upright > 0.9
    finally:
        camera.close()
        env.close()
