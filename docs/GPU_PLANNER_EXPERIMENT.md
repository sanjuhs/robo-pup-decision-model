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

## v3 measured outcome and rejected application gate

v3 trained on RTX 5090 / PyTorch 2.9.0+cu129 in 95.49 seconds (8,000 steps).
New-test first action rose from 38.3% to 96.867%, intent from 58.4% to 97.417%.
Both models passed 23/24 fresh physical trials; both held on an occluded ball.
However the application regression at a settled seed-42 pose returned retrieve
instead of return when held=1. The original assertion was retained. **v3 was not
promoted**. Its checkpoint, all reports, CUDA utilization and confirmed pod
deletion are retained. New GPU estimate $0.1976, about $0.4745 aggregate with
prior GPU usage; this is not an invoice.

## Manual corrective experiment v4 (frozen before retraining)

This is a deliberate second experiment after diagnosis, not an automatic retry.
The user-authorized $30 total still applies. Reuse the same 40-minute/$3 allocation
and deletion monitor, initialize matching tensors from the rejected v3 checkpoint,
and train a new learned decoder from scratch. That decoder takes predicted intent,
energy, held state and slot index. Motor measurements and yaw cannot change its
action logits when predicted intent/needs stay fixed. This is a learned neural
module trained from labels, not a handcoded action table. The transformer still
predicts intent and the RGB classifier grounds object targets.

Use 6,000 new scenes, seed6119, 50% static / 50% 20-step motion frames. Train
8,000 steps, batch128, seed521; initialize from demo/arena-planner-v3. Broaden
question/prefix compositions. Keep v3 test results as development evidence.
Freeze FINAL_TEST4 before this run, exclude those exact strings from training,
and evaluate all 500 new mixed scenes (seed94871) ×12 templates, evaluator seed384.
Compare v2 and v4 on the same data; physical seeds707/808/909 and unchanged
120-second criteria. Also require the original held-state application regression,
all Python checks and browser integration. Retain failed models and trials.

### v4 bootstrap failure and corrected retry

The first v4 pod never trained: its PyPI index requests timed out through all
retries. Its GPU estimate is $0.1033; deletion and absence were verified. No
checkpoint is claimed. The manually corrected retry retains the same frozen
training/evaluation protocol. It removes the index/bootstrap dependency and uses
52 explicit CDN wheel URLs with hashes exported from uv.lock for the tested
Linux x86_64/CPython3.12 image. Every wheel was downloaded and hash-verified
locally before launch. A verified HTTPS CDN preflight fails early on an unusable
host. The provider Torch/CUDA stack is retained and recorded. Raw startup logs
are now bundled even if checkout or installation fails. TLS/hash verification
remains enabled; the default demo remains v2 pending application checks.
