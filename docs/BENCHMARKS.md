# Actual measurements

Measured in the current cloud machine on 2026-10-16 (Asia/Kolkata), Python 3.12.14,
MuJoCo 3.3.7, PyTorch 2.9.0 CPU, four vectorized native environments, one PyTorch
thread. This machine has no GPU. Other training/browser work overlapped some runs;
these are development measurements, not isolated hardware benchmarks.

| Run | Steps | Training seconds | Mean forward speed | Mean return | Falls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Analytic gait | no training | — | 0.129 m/s | 757.1 | 0/5 |
| Curious-v1, initial exploration too large | 131,072 | 92.0 | 0.024 m/s | 438.0 | 0/5 |
| Curious-v2, lower initial exploration | 262,144 | 158.0 | 0.340 m/s | 1136.2 | 0/5 |
| Gentle, target 0.20 m/s | 262,144 | 201.7 | 0.210 m/s | see raw result | 0/5 |
| Zoomies, target 0.55 m/s | 262,144 | 177.0 | 0.510 m/s | see raw result | 0/5 |

Curious runs use a 0.35 m/s target. Gentle/zoomies change speed and energy penalty,
so their returns are not directly comparable. Evaluation: deterministic policy,
seeds 1001–1005, 12 simulated seconds each, randomized initial joint angles and
friction. Five evaluation seeds are not five independently trained policies.
The first run is retained as a failed learning baseline. Reduced Gaussian action
noise (`initial_log_std=-2`) improved the second run; its doubled step budget also
changed, so this is not a controlled single-factor causal experiment.

Speed is body-frame forward velocity; world-X displacement for curious-v2 averaged
2.134 m. Heading drift and lateral motion are not yet penalized. No long-horizon,
rough-terrain, real-world, or statistical generalization claim is made.

Native analytic-gait throughput: **10,000 control steps in 2.803 s = 3,568 steps/s**,
including resets, at 50 Hz control / 200 Hz physics. This is aggregate simulation
throughput, not a training FPS or a GPU result.

The optional MJX backend completed a CPU compiled probe (one world, 50 batched
control steps): **84.63 s compilation**, **9.12 steps/s** after compilation. CPU MJX
is much slower than native MuJoCo for this small batch; that is why it is optional.
GPU throughput and transfer quality will be reported only after the approved run.

Behavior protocol: one deterministic 30-second trial each. These tests have low
coverage and were used to expose controller defects. Initial and revised trials
are retained; the latest controller is rechecked separately. Stop passed and a
nearby approach passed; follow and pushing failed the recorded success criteria.
Pushing can physically move the crate without delivering it to the goal.

The facilitator's five-policy validation queue was interrupted when the user
requested only one additional GPU training run. Partial outputs are not counted
as completed training or evaluation results.

## Cost

Paid resources created so far: none at this measurement checkpoint. CPU development
incurred **$0 in Runpod GPU spend**. This does not estimate the platform's own host
cost. Read-only Runpod access succeeded; the inspected secure-cloud catalog quoted
RTX 5090 at $1.19/hour and H100 SXM at $3.99/hour. Catalog prices are not a billing
receipt, confirmed availability, or storage-inclusive quote. Total GPU ceiling: $30.

## Reproduce

```bash
uv run robopup-benchmark --steps 10000 --out artifacts/recheck-physics.json
uv run robopup-train --participant recheck --run curious --steps 262144
uv run robopup-evaluate --model artifacts/recheck/curious/policy.zip --out artifacts/recheck/curious/evaluation.json
uv run python -m robopup.tasks --model artifacts/recheck/curious/policy.zip --out artifacts/recheck/curious/tasks.json
```

Training is seeded but bitwise reproducibility across platforms is not guaranteed.
The earliest run predates source-hash capture and has weaker provenance; later
runs include source digests. No checkpoint or prior result is overwritten.
