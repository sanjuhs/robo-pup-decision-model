"""Live camera/transformer/executor session, independent of transport and UI."""

import base64
from functools import lru_cache
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from robopup.executor import GoalExecutor, heading
from robopup.scene_planner import TRAINED_VERSION, infer, load_planner
from robopup.vision import PlannerCamera

MODEL_DIR = Path(__file__).resolve().parents[2] / "demo" / "arena-planner"


@lru_cache(maxsize=1)
def trained_model():
    torch.set_num_threads(1)
    return load_planner(MODEL_DIR)


class BrainSession:
    def __init__(self, env):
        self.env = env
        self.executor = GoalExecutor(env)
        self.camera = None
        self.latest_png = None
        self.inference_ms = 0.0

    @property
    def available(self):
        return (MODEL_DIR / "planner.pt").is_file()

    def begin(self, instruction):
        self.executor.begin(instruction)

    def tick(self):
        import time

        ex = self.executor
        if not ex.complete and not ex.failed and self.env.data.time - ex.last_plan >= 0.5:
            if self.camera is None:
                self.camera = PlannerCamera(self.env)
            start = time.perf_counter()
            image = self.camera.capture()
            model, _ = trained_model()
            plan = infer(
                model,
                image,
                ex.instruction,
                self.env._obs(),
                self.env.energy,
                float(self.env.held),
                heading(self.env),
            )
            self.inference_ms = (time.perf_counter() - start) * 1000
            ex.update_plan(plan, image)
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            self.latest_png = base64.b64encode(buffer.getvalue()).decode()
        command = ex.command()
        # Use the evaluated PPO residuals only near their straight-trot training
        # domain. Analytic steering, walk/run/jump/gripping are separate skills.
        learned_motor = (
            self.env.motion == "trot"
            and abs(command[1]) < 0.25
            and self.env.jump_time is None
            and np.linalg.norm(command) > 0.03
        )
        return command, learned_motor

    def status(self):
        return {
            **self.executor.status(),
            "camera_png": self.latest_png,
            "inference_ms": self.inference_ms,
            "model": TRAINED_VERSION,
            "vision": "learned arena pixel segmentation; orthographic 96x96 RGB",
            "energy": "virtual play-energy, not a battery measurement",
        }

    def close(self):
        if self.camera is not None:
            self.camera.close()
