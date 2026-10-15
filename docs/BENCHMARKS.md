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

The curious-v2 intermediate checkpoint at 131,072 steps was re-evaluated without
further training: mean body-frame speed **0.359 m/s**, 0/5 falls, mean return 1065.5,
mean world-X displacement 3.712 m. This supports the shorter workshop step budget;
it is an intermediate checkpoint from the longer run, not an independent training seed.

The optional MJX backend completed a CPU compiled probe (one world, 50 batched
control steps): **84.63 s compilation**, **9.12 steps/s** after compilation. CPU MJX
is much slower than native MuJoCo for this small batch; that is why it is optional.
GPU throughput and transfer quality will be reported only after the approved run.

Behavior protocol: one deterministic 30-second trial each. These tests have low
coverage and were used to expose controller defects. Initial and revised trials
are retained; the latest controller is rechecked separately. Stop passed and a
nearby approach passed; follow and pushing failed the recorded success criteria.
Pushing can physically move the crate without delivering it to the goal.

Ball missions have separate criteria. In the original ball-command final-distance
trial, the ball rolled away and the approach check failed; that record is retained.
The revised yard adds physical rolling friction and walls, plus contact-based
completion for a boop. In `ball-missions-v2.json`, curious-v2 contacted the ball in
**13.64 s** and reached the +X side in **20.96 s**, without falling. Each is only one
deterministic trial. A live browser trial independently confirmed actual ball contact.
The behind criterion is X > ball X + 0.35 m and |Y − ball Y| < 0.45 m after waypoints.

Ten independent stop-mode WebSocket worlds advanced at a minimum **0.989 times
real time** over 75 frames per session. Five simultaneous browser instances also
received independent advancing states. These short capacity tests do not establish
two-hour reliability or ten simultaneous training jobs.

A later check of the expanded physical yard exposed contact-sensor overhead
(minimum ratio 0.815). A conservative 1 m broad-phase check now skips robot/ball
contact-object traversal when contact is geometrically impossible. The final
standalone ten-session probe measured a minimum **0.986 times real time**; a probe
overlapping one additional browser ball-mission test measured **0.963**. All three
records are retained. The physical ball-contact regression and browser mission
still pass after this optimization.

## RTX 5090 attempt evidence

The first authorized GPU attempt used an RTX 5090 (32,607 MiB reported by
`nvidia-smi`), driver 595.91.07, JAX 0.7.2 and MuJoCo MJX 3.3.7. Its physics probe
measured **10,046 batched control steps/s** across 128 worlds after **81.31 s** of
compilation. PPO then failed at its first episode timeout because JAX host rewards
were read-only. Progress reached 73,728 completed logged steps; there is no final
policy or success claim for that attempt. The defect was fixed with writable
boundary arrays and a regression test. The same experiment was retried.

The first pod was deleted and absence verified. Its 333.68-second monitor interval
at the reported $1.19/hour gives an estimated **$0.1103 GPU charge**, excluding
provider billing adjustments/storage. This is an estimate, not an invoice. Raw
lifecycle, hardware, source hashes, configuration, progress, and failure status
are retained in `benchmarks/runpod-attempt-1/`.

The corrected attempt completed **1,048,576 steps in 121.723 s = 8,614 training
steps/s**. Compilation took **83.055 s**; its 128-world physics probe measured
**11,560 batched control steps/s**. Native CPU evaluation on both the pod and the
development machine returned the same five-seed summary: **0.3368 m/s**, mean return
1061.4, mean world-X displacement **3.656 m**, and **0/5 falls**. The checkpoint and
its source/configuration metadata are retained and bundled with the project.
These use different vectorization, rollout and solver settings from the CPU run;
the FPS figures are not a controlled hardware-only speedup claim.

The GPU policy's first ball-boop trial stopped 0.533 m from the ball without contact.
A stand-off navigation target was insufficient to guarantee foot contact. The
controller now targets the ball itself and reports completion only on an actual MuJoCo contact,
without changing the contact success criterion. Failed intermediate controller
trials remain in the attempt-2 directory. No additional training was used for this correction.

With the final controller, CPU curious-v2 booped the ball in **6.38 s** and went
behind it in **20.96 s**. The GPU-trained policy reached the +X side in **16.84 s**,
but **did not contact the ball within 60 s**, even after reaching 0.199 m away.
The flat-ground policy is not a general interaction skill. Use curious-v2 for the
validated ball-boop demo; keep the GPU policy's failure visible when comparing behaviors.

The corrected pod was deleted after artifact retrieval; absence was verified again
with the CLI. Lifecycle intervals give an aggregate **$0.2769 estimated GPU charge**
for the failed and successful attempts. The billing API returned empty histories
for both IDs at the check time, so a final invoice/storage-inclusive amount is
not yet available. No pod or persistent volume was retained. This is well below
the $30 ceiling, based on the observed rate and elapsed intervals.

The facilitator's five-policy validation queue was interrupted when the user
requested only one additional GPU training run. Partial outputs are not counted
as completed training or evaluation results.

## Cost

CPU development incurred **$0 in Runpod GPU spend**. The two disposable GPU
attempts have an aggregate estimated $0.2769 charge as described above, with final
billing not yet posted. This does not estimate the platform's own development-host
cost. The selected pod response reported $1.19/hour. No H100 was provisioned.
Catalog and elapsed-time estimates are not a storage-inclusive receipt. Total GPU ceiling: $30.

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
