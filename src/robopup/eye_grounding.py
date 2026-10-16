"""Conservative eye-image visibility using the arena's trained pixel classifier.

Only RGB pixels enter inference. The 821 weights are extracted, unchanged, from
the checksum-verified v4 checkpoint; no simulator coordinates, segmentation,
object IDs, colour thresholds, or generated VLM answers are consulted.

This classifier learned the original arena's object colours from overhead
images. Applying it to head-camera pixels is a limited domain transfer, not
general object recognition. A missing confident region means "not grounded in
this view", not proof that an object does not exist. Posterior/area thresholds
are engineering abstention rules, not statistically calibrated probabilities.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

OBJECTS = ("ball", "crate", "home", "beacon")
VERSION = "arena-rgb-transformer-v4"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "demo" / "arena-planner"
TARGET_ACTIONS = {"approach": "ball", "behind": "ball", "fetch": "ball",
                  "push": "crate", "follow": "beacon"}


def _largest_component(mask):
    """Largest 4-connected area, with no optional image-processing dependency."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    largest = 0
    for y, x in np.argwhere(mask):
        if visited[y, x]:
            continue
        visited[y, x] = True
        pending = [(int(y), int(x))]
        area = 0
        while pending:
            row, column = pending.pop()
            area += 1
            for ny, nx in ((row - 1, column), (row + 1, column),
                           (row, column - 1), (row, column + 1)):
                if (0 <= ny < height and 0 <= nx < width
                        and mask[ny, nx] and not visited[ny, nx]):
                    visited[ny, nx] = True
                    pending.append((ny, nx))
        largest = max(largest, area)
    return largest


class EyeGrounding:
    def __init__(self, directory=DEFAULT_MODEL_DIR, *, posterior_threshold=.9,
                 minimum_area_fraction=1 / 2304, minimum_pixels=4):
        if (isinstance(posterior_threshold, bool)
                or not isinstance(posterior_threshold, (int, float))
                or not math.isfinite(posterior_threshold) or not .5 <= posterior_threshold <= 1):
            raise ValueError("Visibility posterior threshold must be between 0.5 and 1")
        if (isinstance(minimum_area_fraction, bool)
                or not isinstance(minimum_area_fraction, (int, float))
                or not math.isfinite(minimum_area_fraction)
                or not 0 < minimum_area_fraction <= 1):
            raise ValueError("Visibility minimum area fraction must be between zero and one")
        if (isinstance(minimum_pixels, bool) or not isinstance(minimum_pixels, int)
                or minimum_pixels < 1):
            raise ValueError("Visibility minimum pixel count must be a positive integer")
        directory = Path(directory)
        path = directory / "planner.pt"
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.checkpoint_sha256 != (directory / "SHA256SUMS").read_text().split()[0]:
            raise ValueError("Eye grounding planner checksum mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != VERSION:
            raise ValueError("Eye grounding requires the evaluated v4 arena checkpoint")
        # Reproduce only the released pixel architecture, never the transformer
        # or its overhead projection (which is invalid for perspective images).
        self.pixel = nn.Sequential(
            nn.Conv2d(3, 24, 1), nn.GELU(), nn.Conv2d(24, 24, 1), nn.GELU(), nn.Conv2d(24, 5, 1)
        )
        state = {name.removeprefix("pixel."): tensor for name, tensor in
                 checkpoint["state_dict"].items() if name.startswith("pixel.")}
        self.pixel.load_state_dict(state, strict=True)
        self.pixel.eval().requires_grad_(False)
        self.posterior_threshold = float(posterior_threshold)
        self.minimum_area_fraction = float(minimum_area_fraction)
        self.minimum_pixels = minimum_pixels

    def observe(self, rgb):
        """Return JSON-safe visibility evidence; no object coordinates are inferred."""
        report, _ = self.observe_with_mask(rgb)
        return report

    def observe_with_mask(self, rgb):
        """Also return a uint8 confident-class mask for inspectable diagnostics.

        Labels 1..4 mean ball/crate/home/beacon. Zero means unqualified pixels,
        which includes background and uncertain classifier predictions. This
        is a learned prediction, never a simulator segmentation mask.
        """
        if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
                or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 16 or max(rgb.shape[:2]) > 1024):
            raise ValueError("Eye grounding expects a uint8 RGB image with sides from 16 to 1024")
        height, width = rgb.shape[:2]
        frame = np.ascontiguousarray(rgb).copy()
        with torch.inference_mode():
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0) / 255
            probabilities = self.pixel(tensor).softmax(1)[0].numpy()
        winner = probabilities.argmax(0)
        mask = np.zeros((height, width), dtype=np.uint8)
        minimum_area = max(self.minimum_pixels,
                           math.ceil(height * width * self.minimum_area_fraction))
        objects = {}
        for category, name in enumerate(OBJECTS, 1):
            confident = (winner == category) & (probabilities[category] >= self.posterior_threshold)
            mask[confident] = category
            count = int(confident.sum())
            component = _largest_component(confident)
            objects[name] = {
                "visible": bool(component >= minimum_area),
                "confident_pixels": count,
                "largest_component_pixels": component,
                "maximum_posterior": float(probabilities[category].max()),
                "mean_qualified_posterior": (float(probabilities[category][confident].mean())
                                             if count else None),
            }
        report = {
            "source": "learned arena pixel classifier applied to eye RGB",
            "checkpoint_version": VERSION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "trained_parameters": sum(parameter.numel() for parameter in self.pixel.parameters()),
            "width": width, "height": height,
            "thresholds": {
                "posterior": self.posterior_threshold,
                "minimum_area_fraction": self.minimum_area_fraction,
                "minimum_connected_pixels": minimum_area,
            },
            "objects": objects,
            "visible_objects": [name for name in OBJECTS if objects[name]["visible"]],
            "coordinates_used": False,
            "limitation": "Bounded arena colours; not visible means ungrounded, not nonexistent",
        }
        return report, mask


def require_visible_target(step, report, *, held=False):
    """Gate only the immediate step against the actual activation-frame RGB.

    Call again with a fresh image when activating a later step. Look, turns,
    untargeted motions, and drop are allowed without visual target evidence.
    Held-ball proprioception permits fetch to return an already-held ball;
    it does not provide coordinates or evidence for another object.
    """
    action = step.get("action") if isinstance(step, dict) else getattr(step, "action", None)
    target = TARGET_ACTIONS.get(action)
    if target is None or (action == "fetch" and held is True):
        return
    evidence = report.get("objects", {}).get(target, {}) if isinstance(report, dict) else {}
    if evidence.get("visible") is not True:
        raise ValueError(
            f"Cannot start {action}: {target} is not grounded in the current eye image; look or turn first"
        )
