"""Untrained multimodal planning research scaffold; never used by the live arena.

This tests the image/text/state -> horizon tensor contract. Random initialization
does not provide language understanding, visual recognition, or usable skills.
"""
from dataclasses import dataclass

import torch
from torch import nn

SKILLS = ("stop", "walk", "trot", "run", "jump", "approach", "retrieve", "return", "rest", "recover")


@dataclass(frozen=True)
class PlannerConfig:
    image_size: int = 96
    patch_size: int = 16
    state_dim: int = 53  # Current 50 observations + virtual energy, progress, elapsed goal time.
    text_length: int = 96
    history: int = 4
    horizon: int = 6
    width: int = 128
    heads: int = 4
    layers: int = 3


def encode_instruction(text: str, length: int = 96) -> torch.Tensor:
    """UTF-8 byte tokens, zero padding. Tokenization supplies no learned semantics."""
    result = torch.zeros(length, dtype=torch.long)
    values = list(text.encode("utf-8")[:length])
    result[:len(values)] = torch.tensor(values, dtype=torch.long) + 1
    return result


class MultimodalPlanner(nn.Module):
    """Small transformer encoder with learned plan-slot queries, not trained weights."""

    def __init__(self, config: PlannerConfig = PlannerConfig()):
        super().__init__()
        c = self.config = config
        if c.image_size % c.patch_size or c.width % c.heads:
            raise ValueError("Image/patch sizes and width/head count must divide evenly")
        self.patch = nn.Conv2d(3, c.width, c.patch_size, stride=c.patch_size)
        self.text = nn.Embedding(257, c.width, padding_idx=0)
        self.state = nn.Sequential(nn.LayerNorm(c.state_dim), nn.Linear(c.state_dim, c.width))
        self.kind = nn.Embedding(4, c.width)
        patches = (c.image_size // c.patch_size)**2
        self.position = nn.Parameter(torch.zeros(1, c.history*(patches+1)+c.text_length+c.horizon,
                                                 c.width))
        self.queries = nn.Parameter(torch.randn(1, c.horizon, c.width)*.02)
        layer = nn.TransformerEncoderLayer(c.width, c.heads, c.width*4, dropout=0.,
                                           batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(c.width)
        self.skill = nn.Linear(c.width, len(SKILLS))
        self.target = nn.Linear(c.width, 2)
        self.command = nn.Linear(c.width, 2)
        self.duration = nn.Linear(c.width, 1)
        self.outcome = nn.Linear(c.width, 4)  # Success, collision, fall, energy exhaustion logits.

    def forward(self, images: torch.Tensor, text: torch.Tensor,
                states: torch.Tensor) -> dict[str, torch.Tensor]:
        c = self.config
        if images.ndim != 5 or images.shape[2:] != (3, c.image_size, c.image_size):
            raise ValueError("Images must be [batch, history, 3, image_size, image_size]")
        b, t = images.shape[:2]
        if not 1 <= t <= c.history or states.shape != (b, t, c.state_dim):
            raise ValueError("History and state shapes do not match the planner contract")
        if text.shape != (b, c.text_length) or text.dtype != torch.long:
            raise ValueError("Text must be a padded int64 token matrix")
        if not torch.isfinite(images).all() or not torch.isfinite(states).all():
            raise ValueError("Images and state must be finite")
        if (images < 0).any() or (images > 1).any() or (text < 0).any() or (text > 256).any():
            raise ValueError("Images must be in [0,1] and byte tokens in [0,256]")
        visual = self.patch(images.reshape(b*t, 3, c.image_size, c.image_size))
        visual = visual.flatten(2).transpose(1, 2).reshape(b, t, -1, c.width)
        sensory = torch.cat((visual + self.kind.weight[0],
                             (self.state(states) + self.kind.weight[1]).unsqueeze(2)), dim=2)
        sensory = sensory.reshape(b, -1, c.width)
        language = self.text(text) + self.kind.weight[2]
        queries = self.queries.expand(b, -1, -1) + self.kind.weight[3]
        tokens = torch.cat((sensory, language, queries), dim=1)
        padding = torch.cat((torch.zeros((b, sensory.shape[1]), dtype=torch.bool, device=text.device),
                             text == 0,
                             torch.zeros((b, c.horizon), dtype=torch.bool, device=text.device)), dim=1)
        hidden = self.transformer(tokens + self.position[:, :tokens.shape[1]],
                                  src_key_padding_mask=padding)
        plan = self.norm(hidden[:, -c.horizon:])
        command = self.command(plan).tanh()
        return {"skill_logits": self.skill(plan),
                "target_delta_m": self.target(plan).tanh()*5.,
                "forward_mps": command[..., 0]*.6,
                "yaw_rps": command[..., 1]*1.2,
                "duration_s": .25 + self.duration(plan).sigmoid().squeeze(-1)*1.75,
                "outcome_logits": self.outcome(plan)}
