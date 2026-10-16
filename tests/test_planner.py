"""Tensor-contract tests only; these do not establish learned planning ability."""
import pytest
import torch

from robopup.planner import MultimodalPlanner, PlannerConfig, SKILLS, encode_instruction


def test_image_text_state_horizon_and_gradient_contract():
    torch.set_num_threads(1)
    model = MultimodalPlanner()
    images = torch.rand(1, 2, 3, 96, 96)
    text = encode_instruction("fetch the ball").unsqueeze(0)
    state = torch.randn(1, 2, 53)
    result = model(images, text, state)
    assert result["skill_logits"].shape == (1, 6, len(SKILLS))
    assert result["target_delta_m"].shape == (1, 6, 2)
    assert result["outcome_logits"].shape == (1, 6, 4)
    assert torch.all((result["duration_s"] >= .25) & (result["duration_s"] <= 2))
    assert result["forward_mps"].abs().max() <= .6
    assert all(torch.isfinite(value).all() for value in result.values())
    result["skill_logits"].square().sum().backward()
    assert model.patch.weight.grad.abs().sum() > 0
    assert model.state[1].weight.grad.abs().sum() > 0
    assert model.text.weight.grad.abs().sum() > 0


def test_instruction_padding_and_multibyte_text():
    assert encode_instruction("").sum() == 0
    tokens = encode_instruction("🐾", length=6)
    assert tokens[:4].gt(0).all() and tokens[4:].eq(0).all()
    assert len(encode_instruction("long command", length=3)) == 3


def test_rejects_invalid_image_state_and_token_contracts():
    model = MultimodalPlanner()
    images = torch.zeros(1, 1, 3, 96, 96)
    text = encode_instruction("stop").unsqueeze(0)
    states = torch.zeros(1, 1, 53)
    with pytest.raises(ValueError, match="History"):
        model(images, text, states[..., :-1])
    with pytest.raises(ValueError, match="finite"):
        model(images * float("nan"), text, states)
    with pytest.raises(ValueError, match="byte tokens"):
        model(images, text + 257, states)
    with pytest.raises(ValueError, match="divide"):
        MultimodalPlanner(PlannerConfig(width=127))
