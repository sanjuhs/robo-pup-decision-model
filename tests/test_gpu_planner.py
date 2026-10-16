import hashlib
import importlib.util
import json
from pathlib import Path
import re

import torch

from robopup.planner_curriculum import FINAL_TEST, FINAL_TEST4, VALIDATION, training_phrases
from robopup.planner_v3 import ArenaPlannerV3, VERSION
from robopup.scene_planner import INTENTS, load_planner


def test_final_phrases_excluded_and_curriculum_vocabulary_is_complete():
    model = ArenaPlannerV3()
    excluded = set(FINAL_TEST.values()) | set(FINAL_TEST4.values()) | set(VALIDATION.values())
    for intent in INTENTS:
        phrases = training_phrases(intent)
        assert not excluded.intersection(phrases)
        for phrase in phrases:
            assert all(
                word in model.vocabulary for word in re.findall("[a-z0-9]+", phrase.lower())
            ), phrase


def test_versioned_candidate_roundtrip_and_needs_gradient(tmp_path):
    torch.manual_seed(13)
    torch.set_num_threads(1)
    model = ArenaPlannerV3()
    state = torch.randn(1, 53, requires_grad=True)
    output = model(
        torch.zeros(1, 3, 96, 96), model.encode_instruction("fetch the ball")[None], state
    )
    output["skill_logits"].square().mean().backward()
    assert torch.isfinite(state.grad).all() and torch.all(state.grad[0, 50:].abs() > 0)
    path = tmp_path / "planner.pt"
    torch.save(
        {
            "version": VERSION,
            "architecture": {"width": 96, "layers": 2},
            "state_dict": model.state_dict(),
        },
        path,
    )
    (tmp_path / "SHA256SUMS").write_text(
        hashlib.sha256(path.read_bytes()).hexdigest() + "  planner.pt\n"
    )
    (tmp_path / "vocabulary.json").write_text(json.dumps(model.vocabulary))
    loaded, _ = load_planner(tmp_path)
    assert loaded.version == VERSION
    assert (
        loaded.encode_instruction("could you retrieve that ball").tolist()
        == model.encode_instruction("could you retrieve that ball").tolist()
    )


def test_gpu_export_removes_only_torch_and_retains_hashes():
    path = Path(__file__).resolve().parents[1] / "scripts/export_gpu_runtime.py"
    spec = importlib.util.spec_from_file_location("gpu_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = "numpy==2.3.4 \\\n    --hash=sha256:one\ntorch==2.9.0+cpu \\\n    --hash=sha256:two\npillow==12.3.0 \\\n    --hash=sha256:three\n"
    filtered = module.without_cpu_torch(source)
    assert "torch==" not in filtered and "sha256:two" not in filtered
    assert "--hash=sha256:one" in filtered and "--hash=sha256:three" in filtered


def test_v4_decision_ignores_nuisance_motor_values_for_same_predicted_intent():
    from robopup.planner_v4 import ArenaPlannerV4

    torch.manual_seed(21)
    model = ArenaPlannerV4().eval()
    with torch.no_grad():
        model.intent.weight.zero_()
        model.intent.bias.zero_()
        model.intent.bias[3] = 10
        a = torch.randn(1, 53)
        a[:, 50:52] = torch.tensor([1.0, 1.0])
        b = a.clone()
        b[:, :50] *= 100
        b[:, 52] = -1
        image = torch.zeros(1, 3, 96, 96)
        text = model.encode_instruction("fetch the ball")[None]
        first = model(image, text, a)
        second = model(image, text, b)
    torch.testing.assert_close(first["skill_logits"], second["skill_logits"], rtol=0, atol=0)
    torch.testing.assert_close(first["object_logits"], second["object_logits"], rtol=0, atol=0)
