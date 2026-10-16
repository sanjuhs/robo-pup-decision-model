"""RGB-grounded arena transformer. Only bundled, evaluated weights may execute.

The perception model learns pixel classes for this arena's coloured objects.
This is bounded simulated-domain vision, not a pretrained general VLM.
"""

from pathlib import Path
import json
import re

import numpy as np
import torch
from torch import nn

INTENTS = (
    "stop",
    "approach",
    "behind",
    "fetch",
    "follow",
    "push",
    "walk",
    "run",
    "jump",
    "rest",
    "drop",
    "unsupported",
)
PLAN_SKILLS = (
    "stop",
    "approach",
    "behind",
    "retrieve",
    "return",
    "follow",
    "push",
    "walk",
    "run",
    "jump",
    "rest",
    "drop",
)
TRAINED_VERSION = "arena-rgb-transformer-v2"
WORDS = sorted(
    set(
        "stop stay wait halt stand still here go to the ball boop touch eat sniff red approach behind walk get far side of fetch bring home and return collect retrieve me follow beacon along track purple chase push crate box nudge move orange it ring forward take a slowly slow run sprint fast zoomies jump hop do leap up rest break recharge relax have drop release let put down what is weather tell joke hello fly moon color delete files open door please miso hey now for your favorite".split()
    )
)
VOCABULARY = {word: index + 2 for index, word in enumerate(WORDS)}


def encode_scene_instruction(text, length=96):
    tokens = torch.zeros(length, dtype=torch.long)
    words = re.findall(r"[a-z0-9]+", text.lower())[:length]
    tokens[: len(words)] = torch.tensor(
        [VOCABULARY.get(word, 1) for word in words], dtype=torch.long
    )
    return tokens


class ScenePlanner(nn.Module):
    version = TRAINED_VERSION
    vocabulary = VOCABULARY

    def encode_instruction(self, text):
        return encode_scene_instruction(text)

    def encode_state(self, states):
        return self.state(states)

    def __init__(self, width=96, layers=2):
        super().__init__()
        self.width, self.layers = width, layers
        self.pixel = nn.Sequential(
            nn.Conv2d(3, 24, 1), nn.GELU(), nn.Conv2d(24, 24, 1), nn.GELU(), nn.Conv2d(24, 5, 1)
        )
        self.object_encoder = nn.Linear(3, width)
        self.object_type = nn.Embedding(5, width)
        self.text = nn.Embedding(257, width, padding_idx=0)
        self.state = nn.Sequential(nn.LayerNorm(53), nn.Linear(53, width))
        self.position = nn.Parameter(torch.randn(1, 108, width) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, 6, width) * 0.02)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 4, width * 3, dropout=0.0, batch_first=True, activation="gelu"
            ),
            layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(width)
        self.skill = nn.Linear(width, len(PLAN_SKILLS))
        self.object = nn.Linear(width, 5)  # ball, crate, home, beacon, none
        self.offset = nn.Linear(width, 2)
        self.intent = nn.Linear(width, len(INTENTS))

    def perceive(self, images):
        b, _, height, width = images.shape
        logits = self.pixel(images)
        probabilities = logits.softmax(1)
        # Differentiable weighting suppresses tiny background posteriors. The
        # occupancy gate abstains when an object has no confidently seen pixels.
        weights = probabilities[:, 1:] ** 8
        yy, xx = torch.meshgrid(
            torch.arange(height, device=images.device, dtype=images.dtype),
            torch.arange(width, device=images.device, dtype=images.dtype),
            indexing="ij",
        )
        grid = torch.stack(
            (((xx + 0.5) / width - 0.5) * 6, (0.5 - (yy + 0.5) / height) * 6), dim=-1
        )
        count = weights.sum((2, 3)).clamp_min(1e-8)
        xy = torch.einsum("bkhw,hwd->bkd", weights, grid) / count.unsqueeze(-1)
        confident = (probabilities[:, 1:] > 0.6).sum((2, 3)) >= 2
        objects = torch.cat((xy, confident.float().unsqueeze(-1)), dim=-1)
        none = torch.zeros((b, 1, 3), device=images.device, dtype=images.dtype)
        none[..., 2] = 1.0
        return torch.cat((objects, none), dim=1), logits

    def forward(self, images, text, states):
        if images.ndim == 5:
            images = images[:, -1]
        if states.ndim == 3:
            states = states[:, -1]
        if images.shape[1:] != (3, 96, 96) or text.shape[1:] != (96,) or states.shape[1:] != (53,):
            raise ValueError("Expected RGB96, padded 96-word instruction and 53-value state")
        objects, pixels = self.perceive(images)
        b = images.shape[0]
        visual = self.object_encoder(objects) + self.object_type.weight.unsqueeze(0)
        tokens = torch.cat(
            (
                visual,
                self.text(text),
                self.encode_state(states).unsqueeze(1),
                self.queries.expand(b, -1, -1),
            ),
            dim=1,
        )
        padding = torch.cat(
            (
                torch.zeros((b, 5), dtype=torch.bool, device=text.device),
                text == 0,
                torch.zeros((b, 7), dtype=torch.bool, device=text.device),
            ),
            dim=1,
        )
        hidden = self.norm(self.transformer(tokens + self.position, src_key_padding_mask=padding))
        plan = hidden[:, -6:]
        object_logits = self.object(plan)
        delta = self.offset(plan).tanh()
        selected = torch.einsum("bhk,bkd->bhd", object_logits.softmax(-1), objects[..., :2])
        return {
            "skill_logits": self.skill(plan),
            "object_logits": object_logits,
            "target_delta_m": selected + delta,
            "offset_m": delta,
            "intent_logits": self.intent(hidden[:, -1]),
            "objects": objects,
            "pixel_logits": pixels,
        }


def load_planner(directory: Path):
    import hashlib

    expected = (directory / "SHA256SUMS").read_text().split()[0]
    path = directory / "planner.pt"
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Planner checksum mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["version"] == TRAINED_VERSION:
        model = ScenePlanner(**checkpoint["architecture"])
    else:
        from robopup.planner_v3 import ArenaPlannerV3, VERSION

        if checkpoint["version"] != VERSION:
            raise ValueError("Unsupported planner checkpoint")
        model = ArenaPlannerV3(**checkpoint["architecture"])
    vocabulary = directory / "vocabulary.json"
    if vocabulary.exists() and json.loads(vocabulary.read_text()) != model.vocabulary:
        raise ValueError("Planner vocabulary mismatch")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def infer(model, image, instruction, observation, energy=1.0, progress=0.0, yaw=0.0):
    state = np.concatenate((observation, [energy, progress, yaw / np.pi])).astype(np.float32)
    with torch.inference_mode():
        out = model(
            torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255,
            model.encode_instruction(instruction).unsqueeze(0),
            torch.from_numpy(state).unsqueeze(0),
        )
    skills = out["skill_logits"][0].softmax(-1)
    objects = out["object_logits"][0].softmax(-1)
    intent = out["intent_logits"][0].softmax(-1)
    rows = []
    for h in range(6):
        obj = int(objects[h].argmax())
        # Select one perceived object after decoding. Averaging different
        # physical objects would create an ungrounded target between them.
        xy = out["objects"][0, obj, :2] + out["offset_m"][0, h]
        rows.append(
            {
                "skill": PLAN_SKILLS[int(skills[h].argmax())],
                "object": ("ball", "crate", "home", "beacon", "none")[obj],
                "delta": xy.tolist(),
                "object_delta": out["objects"][0, obj, :2].tolist(),
                "duration_s": 1.0,
                "confidence": float(skills[h].max()),
                "visible": bool(out["objects"][0, obj, 2] > 0.5),
            }
        )
    return {
        "intent": INTENTS[int(intent.argmax())],
        "confidence": float(intent.max()),
        "rows": rows,
        "perception": out["objects"][0].tolist(),
    }
