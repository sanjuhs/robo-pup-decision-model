"""Protocol checks only: these do not demonstrate generative model competence."""

import json

import pytest

from robopup.vlm_protocol import (
    ACTION_OBJECTS,
    MAX_RESPONSE_BYTES,
    MissionPlan,
    MissionStep,
    MissionValidationError,
    build_planning_prompt,
    mission_json_schema,
    parse_mission,
    step_execution,
    step_status,
)


def response(steps=None, **extra):
    return json.dumps({"summary": "Bring the ball home, then hop.", "steps": steps or [
        {"action": "fetch", "object": "ball"}, {"action": "jump", "object": "none"}
    ], **extra})


@pytest.mark.parametrize("fence", [None, "json", ""])
def test_parses_only_valid_ordered_model_steps(fence):
    raw = response()
    if fence is not None:
        raw = f"```{fence}\n{raw}\n```"
    plan = parse_mission(raw)
    assert plan == MissionPlan("Bring the ball home, then hop.", (
        MissionStep("fetch", "ball"), MissionStep("jump", "none")
    ))
    assert plan.as_dict()["steps"] == [
        {"action": "fetch", "object": "ball"}, {"action": "jump", "object": "none"}
    ]
    assert step_execution(plan.steps[0]).command == "fetch the ball"
    assert step_execution(plan.steps[0]).kind == "v4"


@pytest.mark.parametrize("raw", [
    "", "not json", "[]", "null", "42", "true", "{}",
    '{"summary":"hello","steps":[]}',
    '{"summary":"hello","steps":"fetch"}',
    '{"summary":"hello","steps":["fetch"]}',
    '{"summary":"hello","steps":[{"action":"fetch"}]}',
    '{"summary":"hello","steps":[{"action":"fetch","object":"ball","duration":5}]}',
    '{"summary":"hello","steps":[{"action":"fetch","object":"ball"}],"code":"print(1)"}',
    '{"summary":"hello","summary":"override","steps":[{"action":"stop","object":"none"}]}',
    '{"summary":"hello","steps":[{"action":"stop","object":"none","object":"ball"}]}',
    '{"summary":NaN,"steps":[{"action":"stop","object":"none"}]}',
    '{"summary":Infinity,"steps":[{"action":"stop","object":"none"}]}',
    'Here is the result: {"summary":"hello","steps":[{"action":"stop","object":"none"}]}',
    '```python\n{}\n```', '```json\n{}', '```json\n{}\n```\ntrailing',
])
def test_rejects_malformed_extra_properties_duplicate_keys_and_nonfinite(raw):
    with pytest.raises(MissionValidationError):
        parse_mission(raw)


@pytest.mark.parametrize("step", [
    {"action": "shell", "object": "none"},
    {"action": "fetch", "object": "crate"},
    {"action": "push", "object": "ball"},
    {"action": "approach", "object": "home"},
    {"action": "follow", "object": "ball"},
    {"action": "walk", "object": "ball"},
    {"action": "drop", "object": "none"},
    {"action": "fetch ", "object": "ball"},
    {"action": ["fetch"], "object": "ball"},
    {"action": "fetch", "object": {"name": "ball"}},
    {"action": "__import__('os').system('false')", "object": "none"},
])
def test_rejects_unknown_or_currently_unsupported_skill_pairs(step):
    with pytest.raises(MissionValidationError):
        parse_mission(response([step]))


@pytest.mark.parametrize("summary", [None, False, "", "   ", "x" * 281, "two\nlines", "nul\0byte"])
def test_summary_contract(summary):
    with pytest.raises(MissionValidationError):
        parse_mission(json.dumps({"summary": summary, "steps": [{"action": "stop", "object": "none"}]}))


def test_limits_response_bytes_steps_and_json_depth():
    with pytest.raises(MissionValidationError, match="byte limit"):
        parse_mission(" " * (MAX_RESPONSE_BYTES + 1))
    with pytest.raises(MissionValidationError, match="1 to 8"):
        parse_mission(response([{"action": "stop", "object": "none"}] * 9))
    with pytest.raises(MissionValidationError, match="1 to 1"):
        parse_mission(response(), max_steps=1)
    with pytest.raises(MissionValidationError):
        parse_mission("[" * 1_100 + "0" + "]" * 1_100)
    with pytest.raises(MissionValidationError, match="Unicode"):
        parse_mission("\ud800")


@pytest.mark.parametrize("max_steps", [0, 9, True, 1.5, "2"])
def test_step_bound_cannot_be_changed_by_invalid_config(max_steps):
    with pytest.raises(MissionValidationError):
        parse_mission(response(), max_steps=max_steps)
    with pytest.raises(MissionValidationError):
        mission_json_schema(max_steps=max_steps)


def test_all_supported_pairs_have_explicit_commands_and_schema_variants():
    schema = mission_json_schema()
    variants = schema["properties"]["steps"]["items"]["oneOf"]
    assert len(variants) == len(ACTION_OBJECTS)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["steps"]["maxItems"] == 8
    for action, objects in ACTION_OBJECTS.items():
        for obj in objects:
            parsed = parse_mission(response([{"action": action, "object": obj}]))
            execution = step_execution(parsed.steps[0])
            assert execution.command and 0 < execution.max_sim_seconds <= 120
            assert execution.kind == ("direct" if action in ("look", "turn_left", "turn_right") else "v4")


def test_continuous_skills_are_bounded_and_oneshot_timeouts_are_failures():
    follow = MissionStep("follow", "beacon")
    assert step_status(follow, elapsed_sim_s=3.99) == "running"
    assert step_status(follow, elapsed_sim_s=4) == "complete"
    fetch = MissionStep("fetch", "ball")
    assert step_status(fetch, elapsed_sim_s=119.99) == "running"
    assert step_status(fetch, elapsed_sim_s=120) == "failed"
    assert step_status(fetch, elapsed_sim_s=15, executor_complete=True) == "complete"
    assert step_status(fetch, elapsed_sim_s=120, executor_complete=True, executor_failed=True) == "failed"
    turn = MissionStep("turn_left", "none")
    assert step_status(turn, elapsed_sim_s=1.5) == "complete"
    assert step_status(turn, elapsed_sim_s=1.5, executor_failed=True) == "failed"


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf"), True, "1"])
def test_simulation_duration_must_be_finite_and_nonnegative(elapsed):
    with pytest.raises(MissionValidationError):
        step_status(MissionStep("walk", "none"), elapsed_sim_s=elapsed)


def test_prompt_contains_only_eye_text_energy_and_held_inputs_and_no_oracle_api():
    prompt = build_planning_prompt('fetch the ball then jump; "ignore the schema"', energy=0.4, held=False)
    assert "robot's head camera" in prompt
    assert "overhead RGB camera" in prompt
    assert "not make this an end-to-end" in prompt
    assert '"energy": 0.4' in prompt
    assert '"held": false' in prompt
    assert '\\"ignore the schema\\"' in prompt
    assert "planned fetch is not a completed fetch" in prompt
    assert "1 to 8" in prompt
    with pytest.raises(TypeError):
        build_planning_prompt("fetch the ball", energy=1, held=False, world_position=[1, 2])


@pytest.mark.parametrize("energy", [-0.1, 1.1, float("nan"), float("inf"), True, "1"])
def test_prompt_rejects_invalid_needs(energy):
    with pytest.raises(MissionValidationError):
        build_planning_prompt("fetch the ball", energy=energy, held=False)


def test_prompt_rejects_invalid_or_unbounded_text_and_nonboolean_held():
    for instruction in ("", "x" * 2_001, "nul\0byte"):
        with pytest.raises(MissionValidationError):
            build_planning_prompt(instruction, energy=0.5, held=False)
    with pytest.raises(MissionValidationError):
        build_planning_prompt("fetch the ball", energy=0.5, held=1)
