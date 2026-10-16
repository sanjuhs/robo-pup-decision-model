# Capped GPU planner experiment

Protocol frozen before candidate training on 2026-10-16. This expands the user's
previous authorization within the same $30 total session ceiling. It does not
require spending the full budget.

The deployed v2 planner has a known weak result: 89.542% first-action accuracy on
its original final test, with unseen fetch wording also weak. The candidate
changes the numeric state encoder so virtual energy/held/yaw are not normalized
across 50 unrelated motor values. It adds compositional command training and
short physical-motion frames. It keeps the trained arena pixel classifier frozen,
reuses compatible v2 transformer weights, and introduces a new needs encoder.
It remains a small supervised imitation planner, not a general VLM or new PPO gait.

## Resource and training bounds

One RTX 5090 at a confirmed rate no greater than $1.50/hour, 40-minute external
deletion monitor, 35-minute whole job timeout and 25-minute training timeout.
Reserve $1 for prior session usage and allocate $3 to this attempt with $0.20
storage headroom. No network volumes or public ports; no account key in the pod.
The monitor retrieves checksummed artifacts, deletes only its pod and verifies
absence. Estimates and receipts remain separate.

6,000 rendered scenes, generator seed 5041, 20 actual analytic movement steps per
scene, 80/20 scene split. Train 8,000 steps, batch 128, seed 419. Candidate has
241,816 parameters; it extends v2 token IDs rather than silently reordering them.
Training phrases exclude all validation and final templates. Validation samples
include extra energy cases around the 0.15 rest threshold; select only by
validation intent + first-skill + object accuracy. Save partial checkpoints and
progress, source/data/initialization hashes, hardware/runtime and GPU utilization.

The CUDA PyTorch build is retained from the pinned provider image tag. Every
non-Torch runtime package uses the project's lock-exported versions and hashes;
full actual versions are recorded. CUDA must be available; CPU fallback is an error.

## Independent local comparison after export/deletion

- Freeze new final templates in `planner_curriculum.py::FINAL_TEST` before training.
  No final-test metric selects the candidate checkpoint or triggers retraining.
- Generate 500 new scene sequences, seed 73921, 20 movement steps per scene.
  Use **all 500** for final evaluation; the generic generator metadata's 80/20
  training split description does not apply to this final-test set.
- Evaluate v2 and v3 on the identical 6,000 scene/command/state samples (12
  templates × 500 scenes, evaluator seed 384). Report intent, first action,
  horizon/object accuracy, offsets, perception and modality interventions.
- The earlier v2 final test becomes development/regression evidence for later
  models. It is not presented again as an untouched v3 final test.
- Canonical physical commands: approach, behind, fetch, push, follow, walk, run,
  jump. Compare both checkpoints on seeds 404, 505, 606, 120-second limits and
  unchanged physical outcome criteria. Record every trial and failure.
- Promotion requires improved final-test first-action accuracy without a decline
  in intent accuracy or physical-task pass count on these trials. Also retain
  camera grounding, held/energy interventions and the complete application checks.
  These small tests do not establish general vision/language or deployment safety.

The current demo remains on the previously validated checkpoint until these
checks finish. Actual results, cost/cleanup and a promotion or rejection decision
will be added after the run; no expected improvement is described as measured.
