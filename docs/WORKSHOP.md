# Two-hour RoboPup workshop

Audience: 5–10 participants, one facilitator, one private shared host. Everyone
can train a distinct policy from fresh network weights. The shared analytic gait
prior is disclosed; this is residual PPO, not learning all locomotion from zero.

## Before the event

1. Run `bash scripts/setup.sh`. Confirm Python and browser tests pass.
2. Run one 262,144-step curious experiment; inspect five-seed evaluation and retain
   a known baseline checkpoint. Read `BENCHMARKS.md` for actual local timings.
3. Start `bash scripts/start.sh`. Give participants private port-forward access.
   Each tab is independent; changing one target does not move another participant's dog.
   The final short ten-session WebSocket probe measured a minimum real-time ratio of 0.986;
   this is not a two-hour soak test. Five simultaneous browser renderers were also tested.
   The new active-neural probe measured minimum 0.819× real time for five sessions
   and 0.571× for ten, because RGB/inference add CPU load. For smoother interaction
   use local backends per participant or fewer simultaneously active neural goals.
4. Give each participant a unique ID and terminal access. This prototype trusts
   participants with the host filesystem; do not use it for mutually untrusted tenants.
5. Avoid running ten optimizers at once. Queue jobs; begin with one training job,
   and test two concurrent jobs if the host has spare CPU. Reserve capacity for the arena.
6. Confirm storage/checkpoint recovery before terminating any rented pod. Never
   assume stopping a training process stops GPU billing.

## Event plan

| Minutes | Activity | Deliverable |
| --- | --- | --- |
| 0–10 | Meet Miso: camera, treat, manual steering, physical crate | One observed success and one failure |
| 10–25 | Read the observation/action/reward and baseline results | Hypothesis about speed, energy, or stability |
| 25–55 | Each participant trains a first policy with a unique seed | Config, logs, checkpoint, five-seed evaluation |
| 55–75 | Change one parameter; train a second isolated run | Two measured results per participant |
| 75–95 | Load policies in the yard; test approach/follow/push | Record behavior failures as well as successes |
| 95–110 | Compare within a common target-speed/reward group | Reproducible experiment table |
| 110–120 | Show-and-tell and export artifacts | One-minute explanation with evidence |

## Participant commands

```bash
# Alice and Bob use separate names, seeds, and output directories.
uv run robopup-train --participant alice --run curious-01 --seed 101 --config configs/curious.json
uv run robopup-train --participant bob --run curious-01 --seed 102 --config configs/curious.json

# Longer local run; 262,144 is the successful baseline's step budget.
uv run robopup-train --participant alice --run curious-02 --seed 101 --steps 262144

# A fair recheck uses the same evaluation target speed and episode protocol.
uv run robopup-evaluate --model artifacts/alice/curious-02/policy.zip --speed .35 --out artifacts/alice/curious-02/recheck.json
```

The shipped curious-v2 checkpoint is available after setup, so the arena can be
demonstrated immediately. Select it and try `go to the ball` or `go behind the ball`.
The neural RGB controller now passed three varied-start trials each for approach,
behind, fetch, push, follow, walk, run and hop. Use the default curious-v2 hybrid
motor for that demo; custom participant motors can behave differently. A command
being recognized does not prove completion. Inspect the goal and camera/plan panel.
Try `fetch the ball` (about 72–78 simulated seconds), `run forward` and `jump`.
Voice is offline bounded English recognition: review the transcript before Send.
A low virtual play-energy value demonstrates rest/recharge. Use the rules brain
as a clearly labelled privileged-state comparison, not a vision baseline.

Participants can copy `configs/curious.json` to a new file and change one variable.
Profiles `gentle` and `zoomies` intentionally use different speeds/energy penalties;
their returns are not directly comparable to curious. Compare fall rate, target
tracking, torque proxy, displacement, and training time together. Five evaluation
seeds from one training seed are not five independent training runs.

## Facilitator queue

```bash
# Dry-run prints five commands and creates nothing.
uv run python scripts/workshop_queue.py --participants 5 --event day-one
# Explicitly run a sequential, fresh-policy queue on this host.
uv run python scripts/workshop_queue.py --participants 5 --event day-one --execute
```

Supports 5–10 participants. Each job has an eight-minute training deadline plus a
ten-minute process timeout. Existing run directories cause a failure; pick a new
event name rather than overwriting data. The deadline bounds training, not billing.
The browser's Experiments tab discovers completed runs when refreshed.

## Completion criteria

- Every participant has a unique config, seed, final checkpoint, and evaluation.
- Compare like with like: same protocol, target speed, reward, and step budget.
- Demonstrate a failure and explain the scope of any success.
- Export the entire participant directory and source revision, not only a screenshot.
- Facilitator reviews active paid resources and storage after the event. The total
  GPU experimentation ceiling is $30; no provisioning is authorized by these commands.
