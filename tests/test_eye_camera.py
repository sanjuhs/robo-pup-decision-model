"""Actual perspective rendering and isolation from the released overhead input."""

from types import SimpleNamespace
from xml.etree import ElementTree

import mujoco
import numpy as np
import pytest

from robopup.model import arena_xml
from robopup.pet import PetEnv
from robopup.vision import EyeCamera, PlannerCamera


def test_eyes_follow_robot_pose_and_render_different_views():
    env = PetEnv()
    env.reset(seed=42)
    camera = EyeCamera(env)
    try:
        image = camera.capture()
        metadata = camera.metadata()
        assert image.shape == (384, 384, 3) and image.dtype == np.uint8
        assert image.std() > 10
        assert metadata["projection"] == "perspective"
        assert metadata["vertical_fov_degrees"] == 90
        assert set(metadata) == {
            "name", "projection", "width", "height", "vertical_fov_degrees", "position",
            "rotation_camera_to_world", "forward", "up", "simulation_time",
        }
        torso_rotation = env.data.xmat[env.torso].reshape(3, 3)
        np.testing.assert_allclose(
            metadata["position"], env.data.xpos[env.torso] + torso_rotation @ [.412, 0, .092]
        )
        assert np.dot(metadata["forward"], [1, 0, 0]) > .95

        env.data.qpos[0] += .75
        mujoco.mj_forward(env.model, env.data)
        translated = camera.capture()
        np.testing.assert_allclose(
            np.array(camera.metadata()["position"]) - metadata["position"], [.75, 0, 0]
        )
        assert np.mean(np.abs(image.astype(float) - translated)) > 2

        env.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        mujoco.mj_forward(env.model, env.data)
        rotated = camera.capture()
        assert np.dot(camera.metadata()["forward"], [0, 1, 0]) > .95
        assert np.mean(np.abs(translated.astype(float) - rotated)) > 2
    finally:
        camera.close()
        env.close()


def test_eyes_see_forward_scene_without_head_occlusion():
    env = PetEnv()
    env.reset(seed=42)
    camera = EyeCamera(env, size=192)
    try:
        assert camera.capture().shape == (192, 192, 3)
        # Segmentation is used only in this test to verify the optical mount.
        # EyeCamera.capture() and metadata() expose RGB and calibration only.
        camera.renderer.enable_segmentation_rendering()
        camera.renderer.update_scene(env.data, camera="robot_eyes")
        segmentation = camera.renderer.render()
        geom_pixels = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        visible = set(segmentation[geom_pixels, 0].tolist())
        for name in ("floor", "crate_geom", "ball_geom"):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) in visible
        for name in ("head", "torso_geom"):
            assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, name) not in visible
    finally:
        camera.close()
        env.close()


def test_robot_eyes_do_not_change_trained_overhead_pixels_or_physics():
    env = PetEnv()
    env.reset(seed=42)
    # Removing only the new inert camera recreates the released model.
    xml = ElementTree.fromstring(arena_xml(obstacles=True, pet=True))
    torso = xml.find("./worldbody/body[@name='torso']")
    torso.remove(torso.find("camera[@name='robot_eyes']"))
    legacy_model = mujoco.MjModel.from_xml_string(ElementTree.tostring(xml, encoding="unicode"))
    legacy = SimpleNamespace(model=legacy_model, data=mujoco.MjData(legacy_model))
    for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "eq_active"):
        getattr(legacy.data, name)[:] = getattr(env.data, name)
    mujoco.mj_forward(legacy.model, legacy.data)
    overhead = PlannerCamera(env)
    original = PlannerCamera(legacy)
    eyes = EyeCamera(env)
    try:
        original_rgb, original_labels = original.capture(labels=True)
        eyes.capture()
        eyes.metadata()
        current_rgb, current_labels = overhead.capture(labels=True)
        np.testing.assert_array_equal(current_rgb, original_rgb)
        np.testing.assert_array_equal(current_labels, original_labels)
        assert env.model.nq == legacy.model.nq and env.model.nu == legacy.model.nu
        for _ in range(8):
            mujoco.mj_step(env.model, env.data)
            mujoco.mj_step(legacy.model, legacy.data)
        np.testing.assert_array_equal(env.data.qpos, legacy.data.qpos)
        np.testing.assert_array_equal(env.data.qvel, legacy.data.qvel)
    finally:
        overhead.close()
        original.close()
        eyes.close()
        env.close()


@pytest.mark.parametrize("size", [True, 0, 31, 1025, 96.5])
def test_invalid_eye_resolution_is_rejected(size):
    env = PetEnv()
    try:
        with pytest.raises(ValueError, match="size"):
            EyeCamera(env, size=size)
    finally:
        env.close()
