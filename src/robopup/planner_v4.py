"""Candidate with a learned task/needs decoder invariant to nuisance motor values.

This decoder is trained from grammar labels; it is not a hardcoded action table.
RGB supplies object centroids/visibility. The transformer predicts intent; the
neural decoder receives that discrete intent, energy, held state and slot index.
"""

import torch
from torch import nn
import torch.nn.functional as F

from robopup.planner_v3 import ArenaPlannerV3
from robopup.scene_planner import INTENTS, PLAN_SKILLS

VERSION = "arena-rgb-transformer-v4"


class ArenaPlannerV4(ArenaPlannerV3):
    version = VERSION

    def __init__(self, width=96, layers=2):
        super().__init__(width=width, layers=layers)
        del self.skill, self.object, self.offset
        self.decision = nn.Sequential(
            nn.Linear(len(INTENTS) + 2 + 6, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, len(PLAN_SKILLS) + 5 + 2),
        )

    def decode_plan(self, plan, hidden, objects, pixels, states):
        intent = self.intent(hidden[:, -1])
        task = F.one_hot(intent.argmax(-1), num_classes=len(INTENTS)).float()
        inputs = torch.cat((task, states[:, 50:52]), dim=-1)[:, None].expand(-1, 6, -1)
        slots = torch.eye(6, device=states.device, dtype=states.dtype)[None].expand(
            len(states), -1, -1
        )
        decoded = self.decision(torch.cat((inputs, slots), dim=-1))
        skills = decoded[:, :, : len(PLAN_SKILLS)]
        obj = decoded[:, :, len(PLAN_SKILLS) : len(PLAN_SKILLS) + 5]
        offsets = decoded[:, :, -2:].tanh()
        selected = torch.einsum("bhk,bkd->bhd", obj.softmax(-1), objects[..., :2])
        return {
            "skill_logits": skills,
            "object_logits": obj,
            "offset_m": offsets,
            "target_delta_m": selected + offsets,
            "intent_logits": intent,
            "objects": objects,
            "pixel_logits": pixels,
        }
