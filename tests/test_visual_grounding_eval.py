"""Counterfactual scoring failures must remain visible, even with valid JSON."""

import json

import pytest

from robopup.visual_grounding_eval import decision, score_guard, score_pairs, summarize_guard
from robopup.vlm_protocol import MissionValidationError


def response(*actions):
    return json.dumps({"summary": "test", "steps": [
        {"action": action, "object": "ball" if action in ("fetch", "approach") else "none"}
        for action in actions]})


def test_look_preamble_does_not_hide_unsupported_fetch_or_fake_visual_change():
    raw = response("look", "fetch", "stop")
    assert decision(raw, "presence")["choice"] == "fetch"
    pairs = [{"id": "missing", "category": "missing_object", "cases": [
        {"id": "visible", "expected_decision": "fetch"},
        {"id": "absent", "expected_decision": "look"}]}]
    records = [{"id": identifier, "decision": decision(raw, "presence")}
               for identifier in ("visible", "absent")]
    scores = score_pairs(pairs, records)
    assert scores["correct_cases"] == 1
    assert scores["decision_changed_pairs"] == scores["both_correct_pairs"] == 0
    # A parser failure/missing output must not shrink the denominator.
    scores = score_pairs(pairs, records[:1])
    assert scores["case_count"] == 2
    assert scores["valid_pairs"] == scores["both_correct_pairs"] == 0


def test_position_change_must_change_direction_and_be_correct_on_both_sides():
    pairs = [{"id": "move", "category": "changed_position", "cases": [
        {"id": "left", "expected_decision": "turn_left"},
        {"id": "right", "expected_decision": "turn_right"}]}]
    rows = [{"id": identifier, "decision": decision(response(action, "stop"), "position")}
            for identifier, action in (("left", "turn_left"), ("right", "turn_right"))]
    assert score_pairs(pairs, rows)["both_correct_pairs"] == 1
    assert decision(response("turn_left", "turn_right", "stop"), "position")["choice"] == (
        "conflict")
    assert decision(response("stop"), "presence")["choice"] == "other"
    with pytest.raises(MissionValidationError):
        decision("not JSON", "presence")


@pytest.mark.parametrize("visible,truth,outcome", [
    (True, True, "correct_accept"), (True, False, "unsafe_accept"),
    (False, True, "false_abstention"), (False, False, "correct_abstention")])
def test_existing_guard_confusion_outcomes_are_independent_of_model(visible, truth, outcome):
    report = {"objects": {"ball": {"visible": visible}}}
    result = score_guard(report, truth_visible=truth)
    assert result["outcome"] == outcome
    summary = summarize_guard([{"guard": result}])
    assert summary[outcome] == 1 and sum(summary.values()) == 1


def test_held_fetch_exception_preserves_existing_guard_contract():
    assert score_guard({}, truth_visible=False, held=True)["outcome"] == "correct_accept"


def test_shared_baseline_is_not_double_counted_and_conflicting_truth_is_rejected():
    pairs = [{"id": str(index), "category": "missing", "cases": [
        {"id": "base", "expected_decision": "fetch"},
        {"id": str(index), "expected_decision": "look"}]} for index in range(2)]
    rows = [{"id": "base", "decision": {"choice": "fetch"}}]
    scores = score_pairs(pairs, rows)
    assert scores["case_count"] == 3 and scores["correct_cases"] == 1
    pairs[1]["cases"][0]["expected_decision"] = "look"
    with pytest.raises(ValueError, match="consistent truth"):
        score_pairs(pairs, rows)


def test_counterfactual_runner_sends_only_pixels_prompt_and_fixed_state(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace

    from PIL import Image

    from robopup.vlm_runtime import VLMRuntimeError

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("visual_eval_script",
                                                scripts / "evaluate_visual_grounding.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    path = tmp_path / "eye.png"
    Image.new("RGB", (32, 32), "black").save(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = [{"id": name, "image": path.name, "sha256": checksum,
              "expected_decision": choice, "guard_report": {"private_truth": True}}
             for name, choice in (("visible", "fetch"), ("missing", "look"))]
    pairs = [{"id": "unit-fixture", "category": "unit-fixture", "task": "presence",
              "instruction": "same text", "energy": .9, "held": False, "cases": cases}]

    class ControlledRuntime:
        def __init__(self):
            self.calls = 0

        def generate(self, image, instruction, energy, held):
            assert image.size == (32, 32)
            assert (instruction, energy, held) == ("same text", .9, False)
            self.calls += 1
            if self.calls == 2:
                raise VLMRuntimeError("Controlled failure; no model inference")
            return SimpleNamespace(raw_text=response("fetch", "stop"),
                                   as_dict=lambda: {"fixture": True,
                                                    "raw_text": response("fetch", "stop")})

        def status(self):
            return {"fixture": True}

    rows = runner.run_model(pairs, tmp_path, ControlledRuntime())
    assert len(rows) == 2 and "error" in rows[1]
    saved = json.loads((tmp_path / "model-results.json").read_text())
    assert saved["summary"]["case_count"] == 2
    assert saved["summary"]["correct_cases"] == 1
    assert saved["summary"]["valid_pairs"] == 0
    (tmp_path / "pairs.json").write_text(json.dumps(pairs))
    replayed = runner.rescore_recorded_run(pairs, tmp_path)
    assert not replayed["new_inference"] and replayed["summary"]["correct_cases"] == 1
    changed = json.loads(json.dumps(pairs))
    changed[0]["instruction"] = "different task"
    with pytest.raises(ValueError, match="inputs do not match"):
        runner.rescore_recorded_run(changed, tmp_path)
    path.write_bytes(b"tampered image")
    with pytest.raises(ValueError, match="checksum"):
        runner.run_model(pairs, tmp_path, ControlledRuntime())


def test_changed_but_incorrect_plans_are_reported_separately_from_semantic_choices():
    pairs = [{"id": "move", "category": "changed_position", "cases": [
        {"id": "left", "expected_decision": "turn_left"},
        {"id": "right", "expected_decision": "turn_right"}]}]
    rows = [{"id": identifier, "decision": decision(response(*actions), "position")}
            for identifier, actions in (("left", ("turn_left", "turn_right")),
                                        ("right", ("turn_right", "turn_left")))]
    summary = score_pairs(pairs, rows)
    assert summary["executable_plan_changed_pairs"] == 1
    assert summary["decision_changed_pairs"] == summary["both_correct_pairs"] == 0


def test_semantic_decision_is_retained_even_when_instruction_contract_fails():
    observed = decision(response("approach", "fetch", "stop"), "presence")
    assert observed["choice"] == "fetch" and not observed["instruction_compliant"]
    pairs = [{"id": "p", "category": "presence", "cases": [
        {"id": "a", "expected_decision": "fetch"},
        {"id": "b", "expected_decision": "look"}]}]
    assert score_pairs(pairs, [{"id": "a", "decision": observed}])["correct_cases"] == 0
