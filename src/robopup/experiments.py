"""Validated experiment configuration and isolated participant output paths."""
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = Field(default=7, ge=0)
    total_steps: int = Field(default=131072, ge=1024, le=20_000_000)
    n_envs: int = Field(default=4, ge=1, le=16)
    target_speed: float = Field(default=.35, ge=.05, le=.8)
    energy_weight: float = Field(default=.002, ge=0, le=.1)
    learning_rate: float = Field(default=.0003, gt=0, le=.01)
    initial_log_std: float = Field(default=-2., ge=-4, le=0)
    device: Literal["cpu", "cuda"] = "cpu"


def participant_path(root: Path, participant: str, run: str) -> Path:
    for value in (participant, run):
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}", value):
            raise ValueError("Participant/run IDs must be 1-48 letters, numbers, _ or -")
    return root / participant / run
