"""RGB-only learned visibility and immediate-action abstention contracts."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from robopup.eye_grounding import EyeGrounding, require_visible_target
from robopup.pet import PetEnv
from robopup.scene_planner import load_planner
from robopup.vision import EyeCamera, PlannerCamera
from robopup.vlm_protocol import MissionStep

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def grounding():
    torch.set_num_threads(1)
    return EyeGrounding(ROOT / "demo/arena-planner")


def test_eye_pixels_ground_real_objects_but_abstain_on_blank_and_turned(grounding):
    env = PetEnv()
    env.reset(seed=42)
    camera = EyeCamera(env)
    try:
        image = camera.capture()
        actual, mask = grounding.observe_with_mask(image)
        assert actual["visible_objects"] == ["ball", "crate", "beacon"]
        assert actual["trained_parameters"] == 821
        assert actual["coordinates_used"] is False
        assert mask.shape == image.shape[:2] and mask.dtype == np.uint8
        assert actual["thresholds"]["minimum_connected_pixels"] == 64
        blank = grounding.observe(np.zeros_like(image))
        assert blank["visible_objects"] == []
        env.data.qpos[3:7] = [0, 0, 0, 1]  # Actual physical camera turns 180 degrees.
        mujoco.mj_forward(env.model, env.data)
        assert grounding.observe(camera.capture())["visible_objects"] == []
        require_visible_target(MissionStep("fetch", "ball"), actual)
        with pytest.raises(ValueError, match="not grounded"):
            require_visible_target(MissionStep("fetch", "ball"), blank)
    finally:
        camera.close()
        env.close()


def test_pixel_weights_are_exact_release_weights_and_do_not_change_overhead(grounding):
    model, _ = load_planner(ROOT / "demo/arena-planner")
    assert set(model.pixel.state_dict()) == set(grounding.pixel.state_dict())
    for name, value in model.pixel.state_dict().items():
        torch.testing.assert_close(grounding.pixel.state_dict()[name], value, rtol=0, atol=0)
    env = PetEnv()
    env.reset(seed=42)
    overhead, eyes = PlannerCamera(env), EyeCamera(env)
    try:
        before = overhead.capture()
        position = env.data.qpos.copy()
        grounding.observe(eyes.capture())
        np.testing.assert_array_equal(env.data.qpos, position)
        np.testing.assert_array_equal(overhead.capture(), before)
    finally:
        overhead.close()
        eyes.close()
        env.close()


def test_only_current_target_needs_visibility_and_held_ball_can_return():
    blank = {"objects": {}}
    for action in ("look", "turn_left", "turn_right", "walk", "run", "jump", "rest", "stop"):
        require_visible_target(MissionStep(action, "none"), blank)
    require_visible_target(MissionStep("drop", "ball"), blank)
    require_visible_target(MissionStep("fetch", "ball"), blank, held=True)
    for action, target in (("approach", "ball"), ("behind", "ball"), ("fetch", "ball"),
                           ("push", "crate"), ("follow", "beacon")):
        with pytest.raises(ValueError, match="look or turn first"):
            require_visible_target(MissionStep(action, target), blank)
    with pytest.raises(ValueError):
        require_visible_target(MissionStep("push", "crate"), blank, held=True)


@pytest.mark.parametrize("image", [np.zeros((32, 32), dtype=np.uint8),
                                   np.zeros((32, 32, 3)), np.zeros((10, 32, 3), dtype=np.uint8),
                                   np.zeros((32, 32, 4), dtype=np.uint8), "not pixels"])
def test_bad_camera_input_is_rejected(grounding, image):
    with pytest.raises(ValueError, match="RGB"):
        grounding.observe(image)


@pytest.mark.parametrize("options", [{"posterior_threshold": float("nan")},
                                     {"posterior_threshold": .2}, {"minimum_area_fraction": 0},
                                     {"minimum_pixels": True}])
def test_invalid_visibility_thresholds_are_rejected(options):
    with pytest.raises(ValueError):
        EyeGrounding(**options)
