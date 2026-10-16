import mujoco
import numpy as np

from robopup.pet import PetEnv
from robopup.vision import PlannerCamera


def settle(env):
    env.reset(seed=7)
    env.command[:] = 0
    for _ in range(25):
        env.step(np.zeros(12))


def test_contact_gripper_lifts_and_releases_without_pose_teleport():
    env = PetEnv()
    settle(env)
    env.grip(True)
    env._grasp_at_contact()
    assert not env.held  # Asking from far away cannot attach the ball.
    ball_joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_adr = env.model.jnt_qposadr[ball_joint]
    env.data.qpos[env.model.jnt_qposadr[env.lift_joint]] = 0
    env.data.qpos[ball_adr : ball_adr + 3] = [env.data.qpos[0] + 0.57, 0, 0.125]
    mujoco.mj_forward(env.model, env.data)
    before = env.data.qpos.copy()
    env._grasp_at_contact()
    assert env.held and env.grasp_count == 1
    np.testing.assert_array_equal(env.data.qpos, before)
    for _ in range(60):
        env.step(np.zeros(12))
    assert env.data.xpos[env.ball, 2] > 0.25
    env.grip(False)
    for _ in range(100):
        env.step(np.zeros(12))
    assert not env.held and 0.11 < env.data.xpos[env.ball, 2] < 0.14


def test_motor_driven_hop_has_flight_and_lands_without_external_force():
    env = PetEnv()
    settle(env)
    assert env.start_jump()
    assert not env.start_jump()  # No retrigger during flight.
    for _ in range(110):
        env.step(np.zeros(12))
    assert env.jump_flight_seconds > 0.1
    assert env.jump_max_height > 0.43
    assert env.jump_time is None and env.upright > 0.9 and len(env.foot_contacts) >= 2
    assert not np.any(env.data.xfrc_applied) and not np.any(env.data.qfrc_applied)
    assert np.all(env.model.actuator_forcerange[:12, 1] == 18)


def test_rgb_camera_and_training_only_segmentation():
    env = PetEnv()
    settle(env)
    camera = PlannerCamera(env)
    try:
        image = camera.capture()
        assert image.shape == (96, 96, 3) and image.dtype == np.uint8 and image.std() > 10
        rgb, labels = camera.capture(labels=True)
        assert rgb.shape == image.shape and labels.shape == (96, 96)
        assert all(np.any(labels == category) for category in (1, 2, 3, 4))
    finally:
        camera.close()
