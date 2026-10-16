# Changelog

Changes are committed directly to `main` in small, reviewable groups. This is a
research prototype; the linked evidence describes what was actually tested.

## Unreleased

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
