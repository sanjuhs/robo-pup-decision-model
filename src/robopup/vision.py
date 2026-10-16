"""The planner's RGB camera. Simulator IDs are exposed only to dataset labelling."""

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
