# Changelog

Changes are committed directly to `main` in small, reviewable groups. This is a
research prototype; the linked evidence describes what was actually tested.

## Unreleased

### Corrective mission memory and model hosting

- Preserve the genuine CPU closed-loop failure: Qwen repeatedly reintroduced
  fetch after physical completion. Do not promote that trial as a successful
  fetch/hop mission. Replanning now receives the explicit remaining task list;
  a guard rejects substitution/repetition outside that remembered sequence.
- Add a private, bounded OpenAI-compatible vision endpoint to the native backend
  so a Vercel planner can call the actual locally hosted Qwen model. It requires
  a workshop key and never accepts external image URLs or provider recursion.
- Preserve Runpod's failed allocation HTTP500 and independent zero-pod listing.
  Correct the monitor's API-wait arithmetic: unconfirmed allocation is not
  measured GPU spend. Prepare a smaller, tracked-input bootstrap for manual retry.

### Capped generative GPU validation

- Prepare a single RTX 5090 VLM inference experiment with a 25-minute external
  deletion watchdog, $2 allocation and $1.50/hour maximum inside the $30 total.
- Freeze seven actual-image/command probes before GPU inference. Add a genuine
  closed-loop fetch → hop → stop trial with fresh robot-eye inputs and model
  responses at every replan. Native MuJoCo physics remains on CPU.
- Verify all 65 locked non-Torch runtime wheels locally before allocation.
  The job will retain every response/failure plus hardware, cost and deletion
  evidence. This preparation does not claim a completed GPU run.

### Robot appearance and Vercel service boundaries

- Make the original mechanical robot with visible physical pickup the default;
  retain the puppy skin and exact physical-pose rendering. Fresh cold browser
  suite passes 21/21, including actual fetch and strict pause/camera checks.
- Pin Vite 7.3.7; frozen install, build and npm audit pass with zero advisories.
- Add independently packaged app/web/internal-planner Vercel services. The app
  reads its injected PLANNER_URL binding at runtime; planner has no public route.
  Actual `vercel dev -L` routing and internal binding checks passed locally;
  production deployment still requires account authentication and routing review.
- Add a non-root CPU backend container and pinned minimal serverless dependencies.
  Preserve disk/TLS/import build failures rather than claiming they succeeded.

### Robot eyes and sequential generative missions

- Integrate the public Apache-2.0 Qwen3-VL-2B checkpoint as a genuine generative
  model, with explicit revision-pinned download, local or remote vision runtime,
  raw responses and no silent substitute. First actual BF16 CPU image probe
  generated a valid fetch → hop → stop plan in 127.73 s including load; physical
  VLM mission evaluation is separate and remains pending at this checkpoint.
- Add nonblocking per-step mission execution, fresh-eye replanning after verified
  steps, cancellation and private backend access for externally hosted frontends.

- Add a real 384px perspective camera mounted on the robot, preserving the
  original overhead-camera pixels and physical trajectories exactly.
- Add a strict, bounded JSON mission contract for an external generative VLM.
  Individual fetch/push/hop steps require physical outcome checks; timeouts are
  failures. This contract alone is not evidence of language-model competence.


### GPU-trained v4 default

Hosted [full prototype checks passed](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38878935098)
for release source `0a9beda`: 30 Python and 18 browser/render tests.

- Publish the 246,712-parameter, 1,011,137-byte v4 checkpoint after 30 Python and 18
  browser/render checks, including the unchanged held-state regression and live
  v4 identity assertion. Preserve v2 and rejected v3 checkpoints.
- On the same new 6,000-sample test, improve first action 59.817% → 99.967% and
  intent 91.667% → 100%. Both models pass 24/24 fresh physical tasks. Fetch improves
  to 18–20 simulated seconds; push trials take longer. Retain all raw outcomes.
- Train the v4 transformer on RTX 5090 for 101.13 s; refine only its 6,739-parameter
  decoder on CPU for 87.26 s using grammar validation before final testing.
- Verify all five session pods absent. Follow-up GPU estimate $0.4298; aggregate
  session estimate $0.7068 inside $30. New final invoices/storage reconciliation pending.
- Publish the actual 24-second v4 browser fetch recording and checkpoint/source/
  data lineage; keep failed GPU/bootstrap and preliminary CPU-stage evidence.


### GPU bootstrap reliability

- Preserve the failed v4 bootstrap: PyPI timeouts, no training/checkpoint, pod
  deleted, estimated GPU charge $0.1033. No test or TLS verification was bypassed.
- Verify all 52 locked project runtime wheels locally and export direct CDN URLs
  for the tested Python 3.12 pod ABI. Add fast connectivity preflight and export
  early failure logs before checkout. Prepare a manual retry of the same protocol.

### GPU candidate evidence and manual correction

- v3 completed actual RTX 5090 training in 95.49 seconds; new-test first action
  improved 38.3% →96.867%. Both models passed 23/24 new physical trials.
- Reject v3 as default because the existing settled-pose held-object regression
  failed (retrieve instead of return). Preserve the assertion, checkpoint and
  failed application record; default stays v2. Pod deleted and absence verified.
- Add a v4 neural decoder that keeps task/needs decisions independent of nuisance
  motor measurements, plus mixed static/moving training scenes and a newly frozen
  final test. Prepare a second manually diagnosed, capped experiment inside $30.

### Authorized GPU planner candidate

- Add a versioned 241,816-parameter candidate with explicit needs encoding,
  expanded compositional English curriculum and motion-frame training data.
  Keep the existing trained demo and v2 loading compatible.
- Add CUDA-required training, frozen test templates, source/data/initialization
  hashes, a hash-locked GPU runtime export and a 40-minute deletion monitor.
- The user authorized a further Runpod experiment inside the total $30 ceiling.
  CPU smoke and compatibility checks passed; paid results and deployment decisions
  will be recorded after actual training/evaluation.


### Trained arena brain and voice — 2026-10-16

- Train and bundle a 232,414-parameter RGB/text/state transformer with six plan
  rows, visible camera input and closed-loop physical goal checks. Keep the
  earlier untrained research architecture separately labelled.
- Preserve the failed byte-tokenizer run, final untouched scene/phrase test,
  modality interventions, failed retrieval pilot and all 24 successful final
  physics trials. Final-test intent accuracy 97.875%; first-action 89.542%.
- Verify 26 Python tests, 18 browser/render checks and hosted CI on final UI
  source `5a7f148`; publish the actual, unsped 40-second complete fetch recording.
- Add offline PocketSphinx English voice capture/transcription with transcript
  review, virtual play-energy, finite skill commands and explicit neural/rules
  selection. No remote model service, Hugging Face hosting or new GPU spend.

### Physical pet skills and sensor camera

- Add an optional interactive physics rig with a motor-driven lifting gripper,
  contact-gated magnetic weld, and orthographic RGB sensor. The original training
  rig remains the default for reproducibility.
- Add analytic four-beat movement, bouncing run and powered hop controllers.
  These use actual contacts/actuators; they are not newly learned motor policies.
- Test contact-gated capture without pose teleportation, physical lift/release,
  hop flight/landing with no applied external force, and actual RGB/segmentation.
- Preserve the user-approved $30 total ceiling while completing the expanded
  multimodal goal; no new paid resources were used for these changes.

### Golden puppy — 2026-10-16

- Add an original articulated puppy appearance: rounded golden coat, muzzle,
  glossy eyes, floppy ears, paws, curved tail and collar. The research robot
  remains available in Appearance; switching skins preserves the physical world.
- Batch static surface details by material and generate one small fur texture
  locally; no external character asset or new package is downloaded.
- Test trained walking, appearance isolation and paused puppy expressions, and
  publish an actual browser screenshot, recording and measured render limitations.

### Multimodal research foundation — 2026-10-16

- Add an explicitly untrained 769,149-parameter transformer with RGB-history,
  byte-token text, numerical-state inputs and a six-slot skill-plan output.
  It has no trained checkpoint and cannot control the live arena.
- Test tensor bounds/shapes, invalid input rejection and gradients through all
  three input modalities; these are not learned-task evaluations.
- Publish exact MLP/checkpoint counts, a reproducible inspection command, weight
  locations, RTX 5090 usage details, and the vision/voice/energy/skill roadmap.

### Presentation

- Buffer physical poses by 60 ms and interpolate every body over the same pair
  of MuJoCo snapshots. Rendering never extrapolates robot or object motion.
- Use elapsed-time camera smoothing, follow the rendered robot, and add a
  Close-up control for inspecting the trained gait.
- Add decorative tail wagging and blinking, frozen by simulation pause.
  These expressions do not change the policy, leg motion, or physics.
- Verify interpolation and camera response at simulated 30, 60, and 144 Hz,
  plus real-browser trained walking, follow-camera tracking, and pause behavior.
- Add a reproducible real-browser recorder, a captured trained-walk preview, and
  raw software-rendering measurements, including the observed frame-rate limit.

### Public repository checks

- Add a targeted credential-pattern check for tracked files and optional Git
  history, run it in CI, and test that deleted historical fixtures are detected
  without printing their values. This is not an exhaustive secret detector.
- Emit browser test failures as GitHub check annotations as well as ordinary
  logs, so a restricted development host can inspect the failing assertion.

### Fixed

- Preserve the camera target exactly when OrbitControls reports no change,
  avoiding round-off drift with follow disabled while retaining real user input.
  Keep the strict camera test and add an idle/active-input regression check.

- Keep paused render timestamps bit-for-bit constant across interpolation
  intervals. Weighted sums of identical timestamps previously introduced tiny
  rounding differences and intermittently failed the strict pause assertion.
  Keep the browser assertion and cover the fractional-interval regression.

## 0.1.0 — 2026-10-15

### Initial prototype

- Original MuJoCo quadruped, residual PPO training, five-seed evaluation,
  checkpoints, experiment records, and optional batched MJX physics.
- Interactive Three.js yard, per-tab independent worlds, keyboard control,
  physical ball and crate, rule-based behaviors, and local pet-command grammar.
- Bundled CPU and GPU-trained checkpoints with checksum verification; two-hour
  workshop instructions for 5–10 participants and an upstream license audit.
- One completed RTX 5090 experiment: 1,048,576 steps in 121.7 seconds; five native
  CPU evaluation trials averaged 0.337 m/s with no falls. Failed first attempt
  and both pod cleanup records are preserved.
- CPU curious-v2 completed a physical ball contact in a browser test. Following,
  pushing to a goal, learned recovery, and Jev integration remain incomplete.

See [benchmarks](docs/BENCHMARKS.md), [validation](docs/VALIDATION.md), and
[feature status](docs/STATUS.md) for protocols, failures, cost estimates, and limits.
