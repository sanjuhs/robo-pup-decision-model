# RoboPup Arena

**A little dog. A big experiment.**

An interactive robot-pet lab: real MuJoCo contact physics, a playful Three.js yard,
fresh PPO policies, and inspectable experiment results. Inspired by
[robo-jev](https://github.com/STEERIX-home/robo-jev)'s separation of physical judgment
and motor control. This repository's decision layer currently uses rules.

**Prototype, not a hardware controller.** Native physics runs on CPU. PPO learns
joint-target residuals around an analytic trot, starting with fresh network weights.
No pretrained locomotion weights are required for training. GPU physics, RTX 5090
performance, learned recovery, and Jev integration are not yet validated.

## Start the yard

Requires Linux, Python 3.11–3.13, Node 20.19+ or 22.12+, and `uv`.

```bash
bash scripts/setup.sh
bash scripts/start.sh
```

The server listens on port 8000 on the local machine. Access it with a private
port-forward from your development environment. Do not expose this unauthenticated
research server directly to the public Internet. Each browser tab receives an
independent simulated world. Use one Uvicorn worker.

- Click the ground to drop a treat; select **Find a treat**.
- Drag to orbit, scroll to zoom, or enable **Follow cam**.
- **WASD** steers; **Space** requests a standing pose.
- **Pause** freezes physics; **Reset posture** resets the world, including the crate.
- Select a locally trained policy to compare it with the analytic gait.
- **Experiments** reads actual run results; the UI does not start training or cloud resources.

The physical robot and crate transforms come from MuJoCo. Decorative eyes, ears,
tail, treat beacon, and yard boundary are visual only. Obstacles and the crate
collide physically. The crate is moved only by contact, never a scripted force.

## Train your first policy

```bash
uv run robopup-train --participant alice --run first-trick --config configs/curious.json
uv run robopup-evaluate --model artifacts/alice/first-trick/policy.zip --out artifacts/alice/first-trick/recheck.json
uv run python -m robopup.tasks --model artifacts/alice/first-trick/policy.zip --out artifacts/alice/first-trick/tasks.json
```

Choose `curious`, `gentle`, or `zoomies`; customize a copied JSON config to change
speed, seed, or energy penalty. A participant/run directory is created exclusively:
reusing an ID fails rather than overwriting a previous experiment. Outputs include
configuration, source hashes, metadata, episode logs, PPO progress CSV, intermediate
checkpoints, final weights, training time, and five-seed evaluation JSON.

Training defaults to CPU. A small MLP and CPU physics do not automatically benefit
from a GPU. Run the benchmark before renting compute. No script in this repository
creates paid infrastructure.

## Evidence and workshop

- [Actual benchmark results](docs/BENCHMARKS.md), including the failed first run.
- [Feature status and limitations](docs/STATUS.md).
- [Two-hour workshop for 5–10 participants](docs/WORKSHOP.md).
- [Runpod setup, key handling, and approval plan](docs/RUNPOD.md).
- [Architecture and research path](docs/ARCHITECTURE.md).
- [Upstream license audit](docs/UPSTREAM_AUDIT.md).

```bash
uv run pytest -q
uv run ruff check src tests scripts
uv run robopup-benchmark --out artifacts/physics-benchmark.json
cd web
npm ci --ignore-scripts
npm run build
npm test
```

Browser tests need Chromium. Set `CHROMIUM_PATH` to its executable if it is not at
`/usr/bin/chromium`. They run a real local server and verify actual physics state,
controls, policy loading, mobile layout, and simultaneous independent sessions.

## Development

Use the existing checkout in each isolated cloud task; no Git worktree is needed.
`bash scripts/setup.sh` installs the frozen Python/npm dependencies, builds the
browser, verifies the pinned Runpod CLI checksum, and runs the Python tests.
For frontend development, run the backend and `cd web && npm run dev` separately.
Vite proxies API and WebSocket traffic to the backend on port 8000.

Original code and primitive geometry: [MIT](LICENSE). Dependencies retain their
own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). The project is
named RoboPup Arena; its connected GitHub repository is
`sanjuhs/robo-pup-decision-model`.
