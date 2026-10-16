"""Candidate architecture: preserve explicit needs independently of motor normalization."""

import re

import torch
from torch import nn

from robopup.scene_planner import ScenePlanner, VOCABULARY

VERSION = "arena-rgb-transformer-v3"
# Extend IDs rather than shifting the original trained word embeddings.
EXTRA_WORDS = sorted(
    set(
        "i don t is would can could would you your my this that little good dog puppy kindly back over around towards toward then retrieve fetching fetching running walking hopping jumping there again slowly now straight ahead take fetches carries carry homeward toy sphere cube energy tired sleepy please stop wait remain stand beside behind bring return collect retrieve lets let's need want should red orange purple blue do not don't never".split()
    )
    - set(VOCABULARY)
)
V3_VOCABULARY = {
    **VOCABULARY,
    **{word: len(VOCABULARY) + 2 + i for i, word in enumerate(EXTRA_WORDS)},
}


class ArenaPlannerV3(ScenePlanner):
    version = VERSION
    vocabulary = V3_VOCABULARY

    def __init__(self, width=96, layers=2):
        super().__init__(width=width, layers=layers)
        self.state = nn.Sequential(nn.LayerNorm(50), nn.Linear(50, width))
        self.needs = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))

    def encode_state(self, states):
        return self.state(states[:, :50]) + self.needs(states[:, 50:])

    def encode_instruction(self, text):
        tokens = torch.zeros(96, dtype=torch.long)
        words = re.findall(r"[a-z0-9]+", text.lower())[:96]
        tokens[: len(words)] = torch.tensor(
            [self.vocabulary.get(w, 1) for w in words], dtype=torch.long
        )
        return tokens
