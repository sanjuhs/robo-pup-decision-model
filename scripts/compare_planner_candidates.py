#!/usr/bin/env python3
"""Compare frozen, matching planner protocols; report eligibility without deploying."""

import argparse
import json
from pathlib import Path


def compare(baseline, candidate, baseline_physical, candidate_physical):
    for key in ("dataset_sha256", "templates", "samples"):
        if baseline[key] != candidate[key]:
            raise ValueError("Cannot compare different final-test protocols: " + key)

    def outcomes(report):
        return {(t["intent"], t["seed"]): bool(t["success"]) for t in report["trials"]}

    old, new = outcomes(baseline_physical), outcomes(candidate_physical)
    if old.keys() != new.keys():
        raise ValueError("Physical intent/seed protocols differ")
    counts = {
        intent: {
            "baseline": sum(v for (i, _), v in old.items() if i == intent),
            "candidate": sum(v for (i, _), v in new.items() if i == intent),
            "trials": sum(i == intent for i, _ in old),
        }
        for intent in sorted({i for i, _ in old})
    }
    gates = {
        "first_action_improves": candidate["metrics"]["first_skill"]
        > baseline["metrics"]["first_skill"],
        "intent_does_not_decline": candidate["metrics"]["intent"] >= baseline["metrics"]["intent"],
        "physical_success_does_not_decline": sum(new.values()) >= sum(old.values()),
        "no_individual_task_loses_passes": all(
            c["candidate"] >= c["baseline"] for c in counts.values()
        ),
        "rgb_shift_grounded": abs(
            candidate["interventions"]["shifted_rgb"]["measured_x_shift_m"] - 0.5
        )
        < 0.05,
        "blank_rgb_not_visible": not candidate["interventions"]["blank_rgb"]["visible"],
        "low_energy_rest": candidate["interventions"]["low_energy"]["row"]["skill"] == "rest",
        "held_object_returns": candidate["interventions"]["held"]["row"]["skill"] == "return",
    }
    return {
        "eligible_for_application_checks": all(gates.values()),
        "gates": gates,
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": candidate["metrics"],
        "physical_by_task": counts,
        "physical_regressions": [
            {"intent": i, "seed": s} for (i, s), passed in old.items() if passed and not new[i, s]
        ],
        "scope": "Small bounded arena test. Eligibility does not replace application checks or establish general vision/language competence.",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "baseline-physical", "candidate-physical", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    report = compare(
        *(
            json.loads(path.read_text())
            for path in (a.baseline, a.candidate, a.baseline_physical, a.candidate_physical)
        )
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
