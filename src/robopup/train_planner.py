"""Supervised RGB perception + transformer skill plans, with scene/template holdouts."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import platform

import numpy as np
import torch
import torch.nn.functional as F

from robopup.planner_data import HELDOUT, PHRASES, labels
from robopup.scene_planner import (
    INTENTS,
    ScenePlanner,
    TRAINED_VERSION,
    VOCABULARY,
    encode_scene_instruction,
)


def train(data: Path, out: Path, steps=2400, seed=83, device="cpu", perception=None):
    out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "dataset_sha256": hashlib.sha256((data / "scenes.npz").read_bytes()).hexdigest(),
        "seed": seed,
        "steps": steps,
        "device": device,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "perception_sha256": hashlib.sha256((perception / "planner.pt").read_bytes()).hexdigest()
        if perception
        else None,
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "vocabulary.json").write_text(json.dumps(VOCABULARY, indent=2) + "\n")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    source = np.load(data / "scenes.npz")
    images = torch.from_numpy(source["images"]).permute(0, 3, 1, 2).float() / 255
    masks = torch.from_numpy(source["masks"].astype(np.int64))
    obs = source["observations"]
    split = int(len(images) * 0.8)
    model = ScenePlanner().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.001)
    start = time.monotonic()
    log = []
    # Pixel supervision precedes plans so the planner cannot exploit untrained colour centroids.
    pixel_weights = torch.tensor([0.025, 5.0, 5.0, 1.0, 5.0], device=device)
    if perception:
        checkpoint = torch.load(perception / "planner.pt", map_location=device, weights_only=True)
        model.pixel.load_state_dict(
            {
                k.removeprefix("pixel."): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("pixel.")
            }
        )
    for step in range(0 if perception else 450):
        ids = torch.from_numpy(rng.integers(0, split, 32))
        logits = model.pixel(images[ids].to(device))
        loss = F.cross_entropy(logits, masks[ids].to(device), weight=pixel_weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"pixel {step} loss {loss.item():.4f}", flush=True)
    for parameter in model.pixel.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.0008, weight_decay=0.001
    )

    def batch(count, validation=False):
        ids = (
            rng.integers(split, len(images), count) if validation else rng.integers(0, split, count)
        )
        intents = rng.integers(0, len(INTENTS), count)
        energy = rng.uniform(0, 1, count)
        held = rng.integers(0, 2, count)
        state = np.concatenate(
            (obs[ids], energy[:, None], held[:, None], rng.uniform(-1, 1, (count, 1))), axis=1
        ).astype(np.float32)
        phrases = []
        for i in intents:
            key = INTENTS[i]
            phrase = HELDOUT[key] if validation else rng.choice(PHRASES[key])
            phrases.append(
                (
                    rng.choice(["", "miso ", "please ", "hey "])
                    + phrase
                    + rng.choice(["", " please", " now", " for me"])
                )
            )
        token = torch.stack([encode_scene_instruction(p) for p in phrases]).to(device)
        targets = [labels(INTENTS[i], e, h) for i, e, h in zip(intents, energy, held)]
        skills = torch.tensor(np.stack([v[0] for v in targets]), device=device)
        objects = torch.tensor(np.stack([v[1] for v in targets]), device=device)
        offsets = torch.tensor(np.stack([v[2] for v in targets]), device=device)
        return (
            images[ids].to(device),
            token,
            torch.tensor(state, device=device),
            torch.tensor(intents, device=device),
            skills,
            objects,
            offsets,
        )

    for step in range(steps):
        x, text, state, intent, skills, objects, offsets = batch(48)
        result = model(x, text, state)
        loss = (
            F.cross_entropy(result["skill_logits"].flatten(0, 1), skills.flatten())
            + F.cross_entropy(result["object_logits"].flatten(0, 1), objects.flatten())
            + F.cross_entropy(result["intent_logits"], intent)
            + 8 * F.mse_loss(result["offset_m"], offsets)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            row = {"step": step, "loss": float(loss.detach()), "seconds": time.monotonic() - start}
            log.append(row)
            print(json.dumps(row), flush=True)
    model.eval()
    metrics = {
        "intent": [],
        "first_skill": [],
        "horizon_skill": [],
        "object": [],
        "offset_error_m": [],
    }
    confusion = np.zeros((len(INTENTS), len(INTENTS)), dtype=int)
    with torch.inference_mode():
        for _ in range(30):
            x, text, state, intent, skills, objects, offsets = batch(64, True)
            result = model(x, text, state)
            predictions = result["intent_logits"].argmax(-1)
            metrics["intent"].append(float((predictions == intent).float().mean()))
            metrics["first_skill"].append(
                float((result["skill_logits"][:, 0].argmax(-1) == skills[:, 0]).float().mean())
            )
            metrics["horizon_skill"].append(
                float((result["skill_logits"].argmax(-1) == skills).float().mean())
            )
            metrics["object"].append(
                float((result["object_logits"].argmax(-1) == objects).float().mean())
            )
            metrics["offset_error_m"].append(
                float((result["offset_m"] - offsets).norm(dim=-1).mean())
            )
            for a, b in zip(intent.cpu().tolist(), predictions.cpu().tolist()):
                confusion[a, b] += 1
        pixel_confusion = torch.zeros(5, 5, dtype=torch.long)
        for ids in torch.arange(split, len(images)).split(32):
            prediction = model.pixel(images[ids].to(device)).argmax(1).cpu()
            pixel_confusion += torch.bincount(
                (masks[ids] * 5 + prediction).flatten(), minlength=25
            ).reshape(5, 5)
    iou = pixel_confusion.diag() / (
        pixel_confusion.sum(0) + pixel_confusion.sum(1) - pixel_confusion.diag()
    ).clamp_min(1)
    report = {
        "version": TRAINED_VERSION,
        "device": device,
        "seed": seed,
        "steps": steps,
        "training_seconds": time.monotonic() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation_samples": 1920,
        "scene_train_count": split,
        "scene_validation_count": len(images) - split,
        "heldout_instruction_templates": HELDOUT,
        "metrics": {k: float(np.mean(v)) for k, v in metrics.items()},
        "pixel_iou": iou.tolist(),
        "intent_confusion": confusion.tolist(),
        "limitations": "Coloured simulated arena only; holdouts are rendered poses and phrase templates, not general VLM evaluation",
    }
    torch.save(
        {
            "version": TRAINED_VERSION,
            "architecture": {"width": 96, "layers": 2},
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "report": report,
        },
        out / "planner.pt",
    )
    (out / "SHA256SUMS").write_text(
        hashlib.sha256((out / "planner.pt").read_bytes()).hexdigest() + "  planner.pt\n"
    )
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "progress.json").write_text(json.dumps(log, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2400)
    p.add_argument("--seed", type=int, default=83)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--perception", type=Path)
    a = p.parse_args()
    print(json.dumps(train(a.data, a.out, a.steps, a.seed, a.device, a.perception), indent=2))
