"""Versioned candidate training; CUDA required unless explicitly running a CPU smoke test."""

import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from robopup.planner_curriculum import FINAL_TEST, VALIDATION, training_phrases
from robopup.planner_data import labels
from robopup.planner_v3 import ArenaPlannerV3, VERSION
from robopup.scene_planner import INTENTS, PLAN_SKILLS, load_planner


def train(data, out, initialize, steps=8000, batch_size=128, seed=419, device="cuda"):
    out.mkdir(parents=True, exist_ok=False)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required: no CPU fallback for paid experiment")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    metadata = {
        "version": VERSION,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "device": device,
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "dataset_sha256": hashlib.sha256((data / "scenes.npz").read_bytes()).hexdigest(),
        "initialization_sha256": hashlib.sha256(
            (initialize / "planner.pt").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "final_test_templates_frozen_before_training": FINAL_TEST,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "cuda": torch.version.cuda,
        "gpu_capability": list(torch.cuda.get_device_capability()) if device == "cuda" else None,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    model = ArenaPlannerV3().to(device)
    old, _ = load_planner(initialize)
    reusable = {
        k: v
        for k, v in old.state_dict().items()
        if k in model.state_dict()
        and v.shape == model.state_dict()[k].shape
        and not k.startswith("state.")
    }
    model.load_state_dict(reusable, strict=False)
    for p in model.pixel.parameters():
        p.requires_grad_(False)
    source = np.load(data / "scenes.npz")
    images = torch.from_numpy(source["images"]).permute(0, 3, 1, 2).to(device)
    observations = torch.from_numpy(source["observations"]).to(device)
    split = int(len(images) * 0.8)
    if split < 1 or split == len(images):
        raise ValueError("Need training and validation scenes")
    phrases = [training_phrases(intent) for intent in INTENTS]
    tokens = [
        torch.stack([model.encode_instruction(p) for p in bank]).to(device) for bank in phrases
    ]
    valid_tokens = torch.stack([model.encode_instruction(VALIDATION[i]) for i in INTENTS]).to(
        device
    )
    # Labels depend on task, held state and energy. Encode the grammar once;
    # the network must still infer these choices from its multimodal inputs.
    basis = [[labels(i, 1, h) for h in (0, 1)] for i in INTENTS]
    skill_basis = torch.tensor(np.array([[h[0] for h in x] for x in basis]), device=device)
    object_basis = torch.tensor(np.array([[h[1] for h in x] for x in basis]), device=device)
    offset_basis = torch.tensor(np.array([[h[2] for h in x] for x in basis]), device=device)

    def batch(n, validation=False):
        ids = torch.from_numpy(
            rng.integers(split, len(images), n) if validation else rng.integers(0, split, n)
        ).to(device)
        intents = torch.from_numpy(rng.integers(0, len(INTENTS), n)).to(device)
        held = torch.from_numpy(rng.integers(0, 2, n)).to(device)
        energy = torch.from_numpy(
            np.where(rng.random(n) < 0.5, rng.uniform(0.10, 0.20, n), rng.uniform(0, 1, n)).astype(
                np.float32
            )
        ).to(device)
        state = torch.cat(
            (
                observations[ids].clone(),
                energy[:, None],
                held[:, None].float(),
                torch.from_numpy(rng.uniform(-1, 1, (n, 1)).astype(np.float32)).to(device),
            ),
            dim=1,
        )
        if not validation:
            state[:, :50] += torch.randn_like(state[:, :50]) * 0.03
        text = (
            valid_tokens[intents].clone()
            if validation
            else torch.empty(n, 96, dtype=torch.long, device=device)
        )
        if not validation:
            for i, bank in enumerate(tokens):
                where = torch.nonzero(intents == i).flatten()
                text[where] = bank[torch.randint(len(bank), (len(where),), device=device)]
        skills = skill_basis[intents, held].clone()
        objects = object_basis[intents, held].clone()
        offsets = offset_basis[intents, held].clone()
        rest = (
            (energy < 0.15)
            & (intents != INTENTS.index("stop"))
            & (intents != INTENTS.index("drop"))
            & (intents != INTENTS.index("unsupported"))
        )
        skills[rest] = PLAN_SKILLS.index("rest")
        objects[rest] = 4
        offsets[rest] = 0
        return images[ids].float() / 255, text, state, intents, skills, objects, offsets

    def validate():
        model.eval()
        sums = np.zeros(5)
        with torch.inference_mode():
            for _ in range(12):
                x, text, state, intents, skills, objects, offsets = batch(128, True)
                o = model(x, text, state)
                sums += np.array(
                    [
                        float((o["intent_logits"].argmax(-1) == intents).float().mean()),
                        float((o["skill_logits"][:, 0].argmax(-1) == skills[:, 0]).float().mean()),
                        float((o["skill_logits"].argmax(-1) == skills).float().mean()),
                        float((o["object_logits"].argmax(-1) == objects).float().mean()),
                        float((o["offset_m"] - offsets).norm(dim=-1).mean()),
                    ]
                )
        model.train()
        return dict(
            zip(
                ("intent", "first_skill", "horizon_skill", "object", "offset_error_m"),
                (sums / 12).tolist(),
            )
        )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.00025, weight_decay=0.001
    )
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.000015)
    model.train()
    start = time.monotonic()
    progress = []
    best = -1
    best_state = None
    best_metrics = None
    best_step = 0
    for step in range(steps):
        x, text, state, intents, skills, objects, offsets = batch(batch_size)
        o = model(x, text, state)
        loss = (
            F.cross_entropy(o["intent_logits"], intents)
            + 2 * F.cross_entropy(o["skill_logits"][:, 0], skills[:, 0])
            + F.cross_entropy(o["skill_logits"].flatten(0, 1), skills.flatten())
            + F.cross_entropy(o["object_logits"].flatten(0, 1), objects.flatten())
            + 8 * F.mse_loss(o["offset_m"], offsets)
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        schedule.step()
        if step % 250 == 0 or step == steps - 1:
            metrics = validate()
            score = metrics["first_skill"] + metrics["intent"] + metrics["object"]
            if score > best:
                best = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = copy.deepcopy(metrics)
                best_step = step
                torch.save(
                    {
                        "version": VERSION,
                        "architecture": {"width": 96, "layers": 2},
                        "state_dict": best_state,
                        "report": {
                            "steps_completed": step + 1,
                            "incomplete": True,
                            "validation": best_metrics,
                        },
                    },
                    out / "checkpoint.pt",
                )
                (out / "CHECKPOINT_SHA256SUMS").write_text(
                    hashlib.sha256((out / "checkpoint.pt").read_bytes()).hexdigest()
                    + "  checkpoint.pt\n"
                )
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "seconds": time.monotonic() - start,
                "validation": metrics,
            }
            progress.append(row)
            (out / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
            print(json.dumps(row), flush=True)
    if device == "cuda":
        torch.cuda.synchronize()
    report = {
        **metadata,
        "training_seconds": time.monotonic() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_step": best_step,
        "metrics": best_metrics,
        "scene_train_count": split,
        "scene_validation_count": len(images) - split,
        "validation_samples_per_check": 1536,
        "selection": "best sum of validation intent + first skill + object accuracy; final templates unused",
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "perception": "Frozen v2 pixel encoder; shared weights fine-tuned except motor normalization; explicit needs encoder is new",
        "training_phrase_counts": {i: len(p) for i, p in zip(INTENTS, phrases)},
    }
    torch.save(
        {
            "version": VERSION,
            "architecture": {"width": 96, "layers": 2},
            "state_dict": best_state,
            "report": report,
        },
        out / "planner.pt",
    )
    (out / "SHA256SUMS").write_text(
        hashlib.sha256((out / "planner.pt").read_bytes()).hexdigest() + "  planner.pt\n"
    )
    (out / "vocabulary.json").write_text(json.dumps(model.vocabulary, indent=2) + "\n")
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--initialize", type=Path, default=Path("demo/arena-planner"))
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=419)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    a = p.parse_args()
    print(
        json.dumps(
            train(a.data, a.out, a.initialize, a.steps, a.batch_size, a.seed, a.device), indent=2
        )
    )
