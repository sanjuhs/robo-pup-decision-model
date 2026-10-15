# RoboPup Arena contributor instructions

- Use the existing checkout; cloud tasks are already isolated. Do not create a
  worktree unless explicitly requested.
- Never copy secrets from chat or process environment into tracked files, command
  output, documentation, or examples. Runpod uses a secure `RUNPOD_API_KEY` binding.
- Paid infrastructure requires explicit user approval. Total GPU experimentation
  ceiling: $30. Do not provision or start billable resources just to make a test pass.
- Keep changes small and commit related work together. Preserve raw failed runs.
- Required checks: `uv run pytest -q`, `uv run ruff check src tests scripts`,
  `cd web && npm ci --ignore-scripts && npm run build && npm test` when browser code changes.
- Set writable cache directories in restricted cloud environments; see scripts/setup.sh.
- Run representative physics/training evaluations for environment changes.
  A zero-test run, open port, or saved checkpoint does not prove a behavior works.
- Native simulation is CPU; current policies are residual PPO around an analytic
  gait. Stop uses an analytic stance. Reset is not learned self-righting.
- Update docs/STATUS.md and docs/BENCHMARKS.md when evidence or scope changes.
  Do not claim GPU timings, costs, task success, Jev integration, or overnight
  unattended execution without direct evidence.
- No sub-agent delegation is required for this repository.
