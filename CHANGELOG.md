# Changelog

Changes are committed directly to `main` in small, reviewable groups. This is a
research prototype; the linked evidence describes what was actually tested.

## Unreleased

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
