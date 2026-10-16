"""RGB cameras. Simulator object IDs are exposed only to dataset labelling."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

VIEW_METERS = 6.0
OBJECTS = ("ball_geom", "crate_geom", "home_pad", "beacon_geom")


class PlannerCamera:
    def __init__(self, env, size=96):
        self.env = env
        self.renderer = mujoco.Renderer(env.model, height=size, width=size)
        self.camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "planner")
        self.ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in OBJECTS]

    def capture(self, labels=False):
        e = self.env
        e.model.cam_pos[self.camera, :2] = e.data.qpos[:2]
        mujoco.mj_camlight(e.model, e.data)
        self.renderer.update_scene(e.data, camera="planner")
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        image = self.renderer.render().copy()
        if not labels:
            return image
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(e.data, camera="planner")
        segmentation = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for category, geom in enumerate(self.ids, 1):
            mask[
                (segmentation[:, :, 0] == geom)
                & (segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
            ] = category
        return image, mask

    def close(self):
        self.renderer.close()


class EyeCamera:
    """Perspective RGB from the physical robot's forward head mount.

    This camera is independent of the trained planner's overhead camera. Its
    mount follows the simulated torso, including pitch, roll, and yaw. Capture
    metadata describes calibration and the camera pose only; it never includes
    object identities, segmentation, or privileged object coordinates.
    """

    def __init__(self, env, size=384):
        if isinstance(size, bool) or not isinstance(size, int) or not 32 <= size <= 1024:
            raise ValueError("Eye camera size must be an integer between 32 and 1024")
        self.env = env
        self.size = size
        self.camera = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_eyes")
        if self.camera < 0:
            raise ValueError("The robot_eyes camera is missing from the simulation model")
        env.model.vis.global_.offwidth = max(env.model.vis.global_.offwidth, size)
        env.model.vis.global_.offheight = max(env.model.vis.global_.offheight, size)
        self.renderer = mujoco.Renderer(env.model, height=size, width=size)

    def capture(self):
        mujoco.mj_camlight(self.env.model, self.env.data)
        self.renderer.update_scene(self.env.data, camera="robot_eyes")
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        return self.renderer.render().copy()

    def metadata(self):
        """JSON-safe pinhole calibration and camera-to-world extrinsics.

        Rotation columns are camera right, up, and backward in MuJoCo's world
        frame (+Z up); the optical viewing direction is negative camera Z.
        """
        mujoco.mj_camlight(self.env.model, self.env.data)
        rotation = self.env.data.cam_xmat[self.camera].reshape(3, 3)
        return {
            "name": "robot_eyes",
            "projection": "perspective",
            "width": self.size,
            "height": self.size,
            "vertical_fov_degrees": float(self.env.model.cam_fovy[self.camera]),
            "position": self.env.data.cam_xpos[self.camera].tolist(),
            "rotation_camera_to_world": rotation.tolist(),
            "forward": (-rotation[:, 2]).tolist(),
            "up": rotation[:, 1].tolist(),
            "simulation_time": float(self.env.data.time),
        }

    def close(self):
        self.renderer.close()
