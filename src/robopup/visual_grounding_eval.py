"""Offline scoring only: never supplies truth or guard answers to a VLM.

The binary activation gate has no semantic absence verdict. Its rejection is
reported as an abstention; false abstentions and unsafe accepts stay distinct.
"""

from collections import Counter

from robopup.eye_grounding import require_visible_target
from robopup.vlm_protocol import parse_mission


def decision(raw_text, task):
    """Ignore summaries and look preambles, but retain all executable steps."""
    if task not in ("presence", "position"):
        raise ValueError("Unknown visual evaluation task")
    plan = parse_mission(raw_text)
    actions = [step.action for step in plan.steps]
    relevant = {"fetch"} if task == "presence" else {"turn_left", "turn_right", "fetch"}
    choices = [action for action in actions if action in relevant]
    # Multiple contradictory choices must never count as a correct decision.
    choice = choices[0] if len(choices) == 1 else ("conflict" if choices else "look")
    if not choices and not any(action == "look" for action in actions):
        choice = "other"
    allowed = relevant | {"look", "stop"}
    compliant = all(action in allowed for action in actions) and actions[-1] == "stop"
    return {"choice": choice, "instruction_compliant": compliant,
            "steps": [step.as_dict() for step in plan.steps]}


def score_pairs(pairs, records):
    """Missing/invalid generations remain in accuracy denominators."""
    indexed = {row["id"]: row for row in records}
    cases = {}
    for pair in pairs:
        if len(pair["cases"]) != 2:
            raise ValueError("Every counterfactual pair must have two cases")
        for case in pair["cases"]:
            if case["id"] in cases and cases[case["id"]] != case["expected_decision"]:
                raise ValueError("Shared cases require consistent truth")
            cases[case["id"]] = case["expected_decision"]
    results = []
    for pair in pairs:
        rows = [indexed.get(case["id"], {}) for case in pair["cases"]]
        choices = [row.get("decision", {}).get("choice") for row in rows]
        valid = all(choice is not None for choice in choices)
        compliant = all(row.get("decision", {}).get("instruction_compliant", True)
                        for row in rows)
        signatures = [row.get("decision", {}).get("steps") for row in rows]
        results.append({
            "id": pair["id"], "category": pair["category"],
            "valid_both": valid,
            "decision_changed": valid and choices[0] != choices[1],
            "executable_plan_changed": (valid and all(s is not None for s in signatures)
                                        and signatures[0] != signatures[1]),
            "both_decisions_correct": valid and all(choice == case["expected_decision"]
                                                     for choice, case in zip(
                                                         choices, pair["cases"], strict=True)),
            "both_correct": valid and compliant and all(choice == case["expected_decision"]
                                           for choice, case in zip(choices, pair["cases"],
                                                                   strict=True)),
            "choices": choices,
        })
    return {
        "pairs": len(pairs),
        "valid_pairs": sum(row["valid_both"] for row in results),
        "decision_changed_pairs": sum(row["decision_changed"] for row in results),
        "both_correct_pairs": sum(row["both_correct"] for row in results),
        "both_decisions_correct_pairs": sum(row["both_decisions_correct"] for row in results),
        "executable_plan_changed_pairs": sum(row["executable_plan_changed"] for row in results),
        "case_count": len(cases),
        "correct_decision_cases": sum(
            indexed.get(identifier, {}).get("decision", {}).get("choice") == expected
            for identifier, expected in cases.items()),
        "correct_cases": sum(indexed.get(identifier, {}).get("decision", {}).get("choice")
                             == expected and indexed.get(identifier, {}).get("decision", {}).get(
                                 "instruction_compliant", True)
                             for identifier, expected in cases.items()),
        "pair_results": results,
    }


def score_guard(report, *, truth_visible, action="fetch", held=False):
    """Measure the existing gate on a fixed candidate, independent of model output.

    truth_visible is offline renderer evidence (>=64 pixels), NOT an input to
    require_visible_target. This is visibility suitability, not collision safety.
    """
    error = None
    try:
        require_visible_target({"action": action}, report, held=held)
        accepted = True
    except ValueError as exc:
        accepted, error = False, str(exc)
    supported = truth_visible or (action == "fetch" and held is True)
    outcome = (("correct_accept" if supported else "unsafe_accept") if accepted
               else ("false_abstention" if supported else "correct_abstention"))
    return {"gate": "accept" if accepted else "reject_ungrounded",
            "outcome": outcome, "reason": error}


def summarize_guard(records):
    counts = Counter(row["guard"]["outcome"] for row in records)
    return {name: counts[name] for name in
            ("correct_accept", "unsafe_accept", "false_abstention", "correct_abstention")}
