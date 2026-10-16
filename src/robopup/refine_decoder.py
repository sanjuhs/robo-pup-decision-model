"""CPU second stage: balanced grammar-label training for the small v4 needs decoder.

Uses no scene/language final-test results. Vision and transformer remain frozen.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from robopup.planner_data import labels
from robopup.scene_planner import INTENTS, PLAN_SKILLS, load_planner
from robopup.planner_v4 import VERSION


def refine(source, out, steps=10000, seed=811):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    model, parent = load_planner(source)
    if model.version != VERSION:
        raise ValueError("Only v4 has this decoder contract")
    # Exact first-layer reparameterization preserves the initial function while
    # expressing the task's 15% virtual-energy threshold in useful input units.
    with torch.no_grad():
        layer = model.decision[0]
        slope = layer.weight[:, 12].clone() * model.energy_scale
        layer.bias.add_(slope * (0.15 - model.energy_center))
        layer.weight[:, 12].copy_(slope / 20)
        model.energy_center, model.energy_scale = 0.15, 20.0
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.decision.parameters():
        p.requires_grad_(True)
    basis = [[labels(i, 1, h) for h in (0, 1)] for i in INTENTS]
    sb = torch.tensor(np.array([[h[0] for h in x] for x in basis]))
    ob = torch.tensor(np.array([[h[1] for h in x] for x in basis]))
    xb = torch.tensor(np.array([[h[2] for h in x] for x in basis]))

    def targets(task, energy, held):
        skills = sb[task, held].clone()
        objects = ob[task, held].clone()
        offsets = xb[task, held].clone()
        rest = (energy < 0.15) & (task != 0) & (task != 10) & (task != 11)
        skills[rest] = PLAN_SKILLS.index("rest")
        objects[rest] = 4
        offsets[rest] = 0
        return skills, objects, offsets

    def forward(task, energy, held):
        context = model.decision_context(F.one_hot(task, len(INTENTS)).float(), energy, held)
        slots = torch.eye(6)[None].expand(len(task), -1, -1)
        decoded = model.decision(torch.cat((context[:, None].expand(-1, 6, -1), slots), -1))
        return decoded[:, :, :12], decoded[:, :, 12:17], decoded[:, :, 17:].tanh()

    val_energy = torch.tensor(
        [0.025, 0.075, 0.125, 0.145, 0.148, 0.152, 0.155, 0.175, 0.35, 0.75, 0.975]
    )
    val_task = torch.arange(12).repeat_interleave(22)
    val_held = torch.tensor([0, 1]).repeat_interleave(11).repeat(12)
    val_e = val_energy.repeat(24)
    vs, vo, vx = targets(val_task, val_e, val_held)
    optimizer = torch.optim.AdamW(model.decision.parameters(), lr=0.0005, weight_decay=0.0001)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=0.00001)
    weights = torch.tensor([1.0, 1.0, 12.0, 1.0, 0.5])
    start = time.monotonic()
    log = []
    best = -1e9
    chosen = None
    chosen_step = 0
    for step in range(steps):
        task = torch.randint(12, (512,))
        held = torch.randint(2, (512,))
        energy = torch.where(torch.rand(512) < 0.5, 0.13 + torch.rand(512) * 0.04, torch.rand(512))
        skills, objects, offsets = targets(task, energy, held)
        s, o, x = forward(task, energy, held)
        loss = (
            2 * F.cross_entropy(s[:, 0], skills[:, 0])
            + F.cross_entropy(s.flatten(0, 1), skills.flatten())
            + 2 * F.cross_entropy(o[:, 0], objects[:, 0], weight=weights)
            + F.cross_entropy(o.flatten(0, 1), objects.flatten(), weight=weights)
            + 30 * F.mse_loss(x, offsets)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 100 == 0 or step == steps - 1:
            with torch.inference_mode():
                s, o, x = forward(val_task, val_e, val_held)
                metrics = {
                    "first_skill": float((s[:, 0].argmax(-1) == vs[:, 0]).float().mean()),
                    "first_object": float((o[:, 0].argmax(-1) == vo[:, 0]).float().mean()),
                    "horizon_skill": float((s.argmax(-1) == vs).float().mean()),
                    "object": float((o.argmax(-1) == vo).float().mean()),
                    "offset_error_m": float((x - vx).norm(dim=-1).mean()),
                }
                score = (
                    sum(
                        metrics[k]
                        for k in ("first_skill", "first_object", "horizon_skill", "object")
                    )
                    - 10 * metrics["offset_error_m"]
                )
            if score > best:
                best = score
                chosen = {k: v.detach().clone() for k, v in model.decision.state_dict().items()}
                chosen_step = step
                best_metrics = metrics.copy()
            row = {"step": step, "seconds": time.monotonic() - start, "validation": metrics}
            log.append(row)
            print(json.dumps(row), flush=True)
    model.decision.load_state_dict(chosen)
    report = {
        "version": VERSION,
        "parent_gpu_sha256": hashlib.sha256((source / "planner.pt").read_bytes()).hexdigest(),
        "parent_gpu_result": parent["report"],
        "stage": "CPU decoder only, balanced synthetic grammar labels; no scene/text final-test feedback",
        "seed": seed,
        "steps": steps,
        "selected_step": chosen_step,
        "seconds": time.monotonic() - start,
        "decoder_parameters": sum(p.numel() for p in model.decision.parameters()),
        "frozen_parameters": sum(
            p.numel() for n, p in model.named_parameters() if not n.startswith("decision.")
        ),
        "validation_energy_grid": val_energy.tolist(),
        "energy_normalization": {
            "center": 0.15,
            "scale": 20.0,
            "note": "Linear input normalization; decoder remains learned; exact reparameterization before optimization",
        },
        "validation_samples": len(val_task),
        "validation": best_metrics,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    torch.save(
        {
            "version": VERSION,
            "architecture": {
                **parent["architecture"],
                "energy_center": model.energy_center,
                "energy_scale": model.energy_scale,
            },
            "state_dict": model.state_dict(),
            "report": report,
        },
        out / "planner.pt",
    )
    (out / "SHA256SUMS").write_text(
        hashlib.sha256((out / "planner.pt").read_bytes()).hexdigest() + "  planner.pt\n"
    )
    (out / "vocabulary.json").write_text(json.dumps(model.vocabulary, indent=2) + "\n")
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "progress.json").write_text(json.dumps(log, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=10000)
    a = p.parse_args()
    refine(a.source, a.out, a.steps)
