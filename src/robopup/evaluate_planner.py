"""Frozen final test: independent scenes and previously unused command templates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from robopup.planner_data import labels
from robopup.scene_planner import INTENTS, infer, load_planner

TEST_PHRASES = {
    "stop": "miso please wait here",
    "approach": "please go sniff the ball now",
    "behind": "miso go behind the red ball please",
    "fetch": "miso bring me the red ball please",
    "follow": "hey miso chase the purple beacon for me",
    "push": "go nudge that orange crate",
    "walk": "please take a slow walk now",
    "run": "hey miso go run forward",
    "jump": "miso please hop for me",
    "rest": "miso recharge and relax",
    "drop": "miso let go of the red ball",
    "unsupported": "what time is it",
}


def evaluate(model_dir, data_dir, out, templates=None):
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    model, _ = load_planner(model_dir)
    phrases = TEST_PHRASES if templates is None else templates
    source = np.load(data_dir / "scenes.npz")
    rng = np.random.default_rng(384)
    counts = dict(intent=0, first_skill=0, horizon_skill=0, object=0)
    per_intent = {}
    offset_error, count = 0.0, 0
    confusion = torch.zeros(5, 5, dtype=torch.long)
    distances = []
    visible_recall = [0, 0]
    with torch.inference_mode():
        for start in range(0, len(source["images"]), 32):
            raw = source["images"][start : start + 32]
            images = torch.from_numpy(raw).permute(0, 3, 1, 2).float() / 255
            masks = torch.from_numpy(source["masks"][start : start + 32].astype(np.int64))
            objects, pixels = model.perceive(images)
            confusion += torch.bincount(
                (masks * 5 + pixels.argmax(1)).flatten(), minlength=25
            ).reshape(5, 5)
            yy, xx = torch.meshgrid(torch.arange(96), torch.arange(96), indexing="ij")
            for k in range(4):
                mask = masks == k + 1
                area = mask.sum((1, 2))
                visible = area >= 2
                predicted = objects[:, k, 2] > 0.5
                visible_recall[0] += int((visible & predicted).sum())
                visible_recall[1] += int(visible.sum())
                center = torch.stack(
                    (
                        ((mask * xx).sum((1, 2)) / area.clamp_min(1) + 0.5) / 96 * 6 - 3,
                        3 - ((mask * yy).sum((1, 2)) / area.clamp_min(1) + 0.5) / 96 * 6,
                    ),
                    -1,
                )
                distances.extend(
                    (center - objects[:, k, :2]).norm(dim=-1)[visible & predicted].tolist()
                )
            for index, intent in enumerate(INTENTS):
                n = len(raw)
                energy = rng.uniform(0, 1, n)
                held = rng.integers(0, 2, n)
                state = np.concatenate(
                    (
                        source["observations"][start : start + n],
                        energy[:, None],
                        held[:, None],
                        rng.uniform(-1, 1, (n, 1)),
                    ),
                    axis=1,
                ).astype(np.float32)
                expected = [labels(intent, e, h) for e, h in zip(energy, held)]
                skills = torch.tensor(np.stack([v[0] for v in expected]))
                target_objects = torch.tensor(np.stack([v[1] for v in expected]))
                offsets = torch.tensor(np.stack([v[2] for v in expected]))
                result = model(
                    images,
                    model.encode_instruction(phrases[intent]).repeat(n, 1),
                    torch.from_numpy(state),
                )
                correct = int((result["intent_logits"].argmax(-1) == index).sum())
                counts["intent"] += correct
                per_intent.setdefault(intent, [0, 0])
                per_intent[intent][0] += correct
                per_intent[intent][1] += n
                counts["first_skill"] += int(
                    (result["skill_logits"][:, 0].argmax(-1) == skills[:, 0]).sum()
                )
                counts["horizon_skill"] += int((result["skill_logits"].argmax(-1) == skills).sum())
                counts["object"] += int(
                    (result["object_logits"].argmax(-1) == target_objects).sum()
                )
                offset_error += float((result["offset_m"] - offsets).norm(dim=-1).sum())
                count += n
    # Intervene on inputs while holding other modalities fixed.
    image, observation = source["images"][0], source["observations"][0]
    cases = {}
    for name, text, energy, held in [
        ("fetch", "fetch the ball", 1, 0),
        ("push", "push the crate", 1, 0),
        ("low_energy", "fetch the ball", 0.05, 0),
        ("held", "fetch the ball", 1, 1),
    ]:
        p = infer(model, image, text, observation, energy, held)
        cases[name] = {"intent": p["intent"], "row": p["rows"][0]}
    blank = infer(model, np.zeros_like(image), "fetch the ball", observation)
    cases["blank_rgb"] = {"visible": blank["rows"][0]["visible"]}
    p = infer(model, image, "fetch the ball", observation)
    shifted = np.roll(image, 8, axis=1)
    q = infer(model, shifted, "fetch the ball", observation)
    cases["shifted_rgb"] = {
        "expected_x_shift_m": 0.5,
        "measured_x_shift_m": q["rows"][0]["object_delta"][0] - p["rows"][0]["object_delta"][0],
    }
    iou = confusion.diag() / (confusion.sum(0) + confusion.sum(1) - confusion.diag()).clamp_min(1)
    report = {
        "model_sha256": hashlib.sha256((model_dir / "planner.pt").read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256((data_dir / "scenes.npz").read_bytes()).hexdigest(),
        "scenes": len(source["images"]),
        "samples": count,
        "templates": phrases,
        "metrics": {
            k: v / (count * 6 if k in ("horizon_skill", "object") else count)
            for k, v in counts.items()
        },
        "offset_error_m": offset_error / (count * 6),
        "per_intent_accuracy": {k: v[0] / v[1] for k, v in per_intent.items()},
        "pixel_iou": iou.tolist(),
        "visible_object_centroid_mae_m": float(np.mean(distances)),
        "visible_object_recall": visible_recall[0] / visible_recall[1],
        "interventions": cases,
        "scope": "All supplied independent scenes and selected frozen templates; no retraining on this final test. Bounded arena vision/language.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--curriculum", choices=["v2", "v3"], default="v2")
    a = p.parse_args()
    if a.curriculum == "v3":
        from robopup.planner_curriculum import FINAL_TEST

        templates = FINAL_TEST
    else:
        templates = None
    print(json.dumps(evaluate(a.model, a.data, a.out, templates), indent=2))
