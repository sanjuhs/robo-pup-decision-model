# Actual measurements

Measured in the current cloud machine on 2026-10-16 (Asia/Kolkata), Python 3.12.14,
MuJoCo 3.3.7, PyTorch 2.9.0 CPU, four vectorized native environments, one PyTorch
thread. This machine has no GPU. Other training/browser work overlapped some runs;
these are development measurements, not isolated hardware benchmarks.

## Current v4 GPU planner and independent comparison

Raw records: [`gpu-planner-v4/`](../benchmarks/gpu-planner-v4), rejected candidate:
[`gpu-planner-v3/`](../benchmarks/gpu-planner-v3). Both checkpoints see the same
new seed-94871 test: all 500 mixed static/motion poses × 12 frozen new templates,
6,000 task/state combinations, evaluator seed 384. Checkpoint selection used
training validation; these final results did not trigger further training.

| Metric | Prior v2 | Current v4 |
| --- | ---: | ---: |
| Intent accuracy | 91.667% | 100% |
| First-action accuracy | 59.817% | 99.967% (2 errors / 6,000) |
| Horizon skill accuracy | 61.867% | 99.975% |
| Object accuracy | 85.947% | 99.95% |
| Mean relative-offset error | 6.14 cm | 1.15 cm |
| Visible-object centroid MAE vs segmentation | 7.10 cm | 7.10 cm |
| Fresh physical goals | 24/24 | 24/24 |
| Fetch, three matching starts | 74.44–77.36 s | 18.34–19.52 s |
| Push, three matching starts | 11.78–14.78 s | 13.28–31.90 s |
| Behind-ball, three matching starts | 30.14–96.16 s | 30.36–89.68 s |

Physics seeds 707/808/909, 120-second limits and unchanged contact/pose/flight/
return criteria; same CPU curious-v2 motor and analytic skill executor. This is
one trained model per version and three varied starts per activity, not independent
training seeds. Push took longer with v4. The earlier occluded-ball failures and
all rejected-model trials remain visible. Current image classes are fixed coloured
arena objects; the vision encoder was frozen from v2, so no new visual-domain
recognition improvement is claimed.

| Training/lifecycle stage | Actual measurement |
| --- | --- |
| v3 RTX 5090 optimization | 8,000 updates, batch 128, 95.49 s; fresh first action 96.867%, then rejected by settled held-state application test |
| v4 failed bootstrap | PyPI request timeouts before training; no checkpoint, deleted; estimated GPU $0.1033 |
| Corrected v4 RTX 5090 optimization | 8,000 updates, batch 128, 101.13 s; actual Torch 2.9.0+cu129, CUDA 12.9, device capability 12.0 |
| v4 scene generation | 6,000 poses, seed 6119, half static / half after 20 physical steps; 46.93 s |
| CPU decoder stage 1 | 2,500 updates, 18.90 s; validation/object/offset weaknesses retained |
| Final CPU decoder stage | 10,000 updates, 87.26 s; only 6,739 parameters optimized, vision/transformer frozen; selected from grammar-grid validation |
| Follow-up GPU estimates | v3 $0.1976 + failed v4 $0.1033 + successful v4 $0.1290 = **$0.4298** |
| Aggregate session GPU estimate | Prior two attempts $0.2769 + follow-up $0.4298 = **$0.7068** of $30 ceiling |
| Resource cleanup | All five own pods deleted and absent; no network volume or public port created |

GPU estimates use create-to-delete elapsed time at confirmed $1.19/hour. They are
not final receipts; storage reconciliation and new billing posting are pending.
The provider Torch/CUDA stack comes from its image; all 52 project non-Torch
runtime wheels were locally hash-verified and installed from explicit locked URLs.
Hardware, driver, actual packages, utilization and source/data/init hashes are
retained. Native physics and later deployment inference run on CPU. Timing of
these different training recipes is not a controlled hardware-speedup comparison.

Full local application validation passed 30 Python / 18 browser-render tests. The
actual browser v4 fetch recording has 103 captured frames over 24.39 seconds using
software WebGL; device/hardware mobile FPS remains unmeasured.

Current v4 active-neural WebSocket probes advanced all independent worlds for five
simulated seconds: minimum 0.814× real time across five sessions and 0.570× across
ten. These short probes are not a two-hour workshop soak or browser FPS test;
raw session timings are in the same v4 records directory.

## Historical v2 trained multimodal controller (2026-10-16)

Raw reports: [`multimodal-2026-10-16/`](../benchmarks/multimodal-2026-10-16).
This entire extension used native CPU simulation/training and **$0 additional
Runpod usage**. The prior GPU estimate remains $0.2769; no final invoice is claimed.

| Measurement | Actual result |
| --- | --- |
| Training scenes | 1,800 independent rendered poses, 27.97 s generation; 1,440 train / 360 validation |
| Failed byte-tokenizer model | 616.24 s, 2,400 transformer steps plus 450 pixel steps; held-out intent 68.33% |
| Deployed word-tokenizer retry | 411.14 s, 1,600 transformer steps; reused only trained pixel encoder from the failed run |
| Retry validation | 1,920 scene/phrase/state samples; intent 100%, first action 99.219%, horizon skills 98.490%, objects 99.427% |
| Untouched final test | 400 new seed-927 scenes × 12 new phrase compositions = 4,800 samples; intent 97.875%, first action 89.542%, horizon skills 89.25%, object 88.604% |
| Final-test fetch wording | 74.5% intent accuracy; other 11 tested templates 100%. No retraining after this test |
| Visible-object perception | Mean centroid error 7.21 cm against segmentation centroids, visible recall 100%; not exact physical-centre error |
| RGB render + inference | 100 warmed CPU samples, one thread: median 12.53 ms, p95 17.72 ms; excludes PNG/network/browser; overlapped software-browser trial |
| Checkpoint | 232,414 parameters; 946,381 bytes, SHA-256 recorded with release |

The final test includes energy/held-state interventions and random numerical
states. Its composed wording/state failures remain visible; confidence is not
calibrated. The data are static scene poses, not independent physical rollouts.
Two development runs total 1,027.38 CPU training seconds. Architecture was revised
after validation failure, then tested once on the fresh final set. Initial
uncommitted source snapshots were not captured; provenance limits are explicit
in `demo/arena-planner/release.json`.

The **final physical protocol** uses actual RGB/model inference every 0.5 simulated
seconds, the CPU curious-v2 PPO during near-straight trot and analytic steering /
other skills. Seeds 101, 202, 303 randomize initial yaw ±0.12 rad and ball/crate
positions ±0.15 m. They are three varied starts for one trained planner, not three
independently trained policies. Oracle mechanics trials are separate records.

| Activity | Passes | Simulated duration / criterion |
| --- | ---: | --- |
| Ball approach | 3/3 | 5.24–5.50 s; actual robot/ball contact |
| Behind ball | 3/3 | 26.24–34.00 s; +X >0.35 m, Y within 0.45 m |
| Fetch | 3/3 | 71.86–78.16 s; contact, weld capture, lift >0.2 m, return, lower and release on floor within 0.6 m of home |
| Push | 3/3 | 11.78–13.78 s; crate within 0.35 m of (3,0) |
| Follow | 3/3 | 35.02 s each; final beacon error 0.367–0.381 m, upright |
| Walk | 3/3 | 2.42–2.48 s for 0.5 m forward; four-phase analytic schedule |
| Run | 3/3 | 2.74–2.92 s for 1.2 m; 0.28–0.34 s measured all-feet flight |
| Hop | 3/3 | 1.62 s sequence; 0.28–0.30 s flight, max base height 0.463 m, landed upright with ≥2 foot contacts |

All 24 passed with no falls, over 42.49 wall seconds of accelerated headless
simulation. Fetch is slow: turning/alignment dominate. Walking still had
0.02–0.10 s of all-feet flight; call it an analytic four-phase movement, not a
biomechanically validated walking gait. Run/hop are analytic actuator targets,
not newly RL-trained skills. Grasping is a magnetic tool, not an anatomical mouth.

The original RGB pilot fetch failed after release because the ball rolled outside
the unchanged 0.6 m criterion. Its trace is retained. The correction adds settling,
lowering and a gentle release; the criterion was not loosened. A behind-route
latch fixed looping waypoints. Oracle-only successes are not counted as RGB-agent
success. These 24 short tests do not establish arbitrary navigation/manipulation,
real-world transfer, general language/vision or calibrated task success.

## Concurrent neural sessions

A new short probe exercised actual camera/model inference and independent physics,
not just idle worlds. All participants issued fetch and received 125 advancing
state frames each with neural plans/RGB and no fall. Five active neural sessions
measured a minimum **0.819× real time**; ten measured **0.571×** on this CPU host.
This exposes the additional camera/inference load; the old idle-world near-real-time
figure does not describe this path. The test covers only about five simulated
seconds/session, not completion of ten simultaneous fetches or a two-hour soak.
For a smoother workshop, use local backends per participant or fewer concurrently
active neural goals. No extra GPU/server was provisioned to improve this result.

## Historical motor and infrastructure measurements

The following earlier results describe the previous rig/controller revisions.
Their failed follow/push trials are retained and do not contradict the later
RGB-controller results above. Throughput figures below do not benchmark ten
simultaneously active neural planners.

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
The later GPU throughput and transfer recheck are reported below.

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

## Browser presentation capture (2026-10-15)

The bundled `runpod/5090-once` checkpoint was replayed through the real browser
arena, with Close-up/follow camera and forward keyboard input. This was evaluation
of existing weights; no new training or paid resources were used. In **6.018 wall
seconds**, physics advanced **5.760 seconds** and the robot moved **2.012 m along
world X**, without requesting a reset. Later frames encounter the crate; this is
not a push-to-goal success or a flat-ground evaluation protocol.

Chromium used **SwiftShader software WebGL** while recording at a 1440×1050 page
size. The 41 sampled render frames had a **143.9 ms median** and **368.8 ms p95**
frame interval (roughly 7 frames/s during capture). Buffered interpolation is
refresh-rate independent, but this measurement does not demonstrate smooth
30/60 FPS on hardware-accelerated browsers. Recording itself adds overhead.
Raw timing, positions, camera targets, and expression telemetry
are in [the capture record](../benchmarks/presentation-2026-10-15/measurements.json).
The README GIF crops the viewport and loops the first 3.6 seconds at the original
playback speed; repeated GIF frames do not represent extra physics steps. The
linked WebM retains the full recorded page and captured duration.

## Golden-puppy presentation (2026-10-16)

The original procedural skin has **18 physical attachments, 18,248 surface
triangles, 44 mesh draws**, and **382,736 bytes of unique geometry buffers**
(renderer-side buffers, not a model download). One generated 256×256 RGBA bump
texture is shared across coat materials. The browser JS build is about 547 kB
uncompressed / 141 kB gzip, including Three.js. There is no GLB/texture download.

A fresh real-browser replay of the existing GPU checkpoint advanced **5.800
simulation seconds in 6.016 wall seconds**, moved **2.019 m along world X**, and
did not request a reset. Software-WebGL capture sampled **32 render frames**,
with **179.1 ms median** and **536.3 ms p95** frame intervals, about 5 frames/s.
This is a capture on SwiftShader with a larger 993×605 arena viewport, not a
controlled comparison or a hardware/mobile performance claim. Later frames
encounter the crate; no manipulation success is implied. See the
[raw record](../benchmarks/puppy-presentation-2026-10-16/measurements.json).
The screenshot and looped GIF in the README are taken from actual browser output.
No new training or billable resources were used.

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
