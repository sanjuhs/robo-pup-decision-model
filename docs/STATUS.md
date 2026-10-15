# Prototype status

This file separates implemented paths from measured task success. See
`BENCHMARKS.md` and the raw JSON in `benchmarks/` for evidence.

| Capability | Current evidence |
| --- | --- |
| Native headless MuJoCo physics | Tested; physical quadruped, contacts, crate, obstacles and ball |
| PPO from fresh weights | Completed CPU runs; residual gait policies, not end-to-end torque learning |
| First locomotion policy | Curious-v2: 0.340 m/s body-frame forward speed on five 12-second held-out trials; no falls |
| Interactive browser | Tested desktop/mobile; buffered physical geometry, orbit/follow camera, keyboard, pause/reset, policy loading. Cosmetic expressions are separate from learned locomotion. |
| Multiple participants | Five simultaneous independent browser worlds tested; no shared-world collaboration |
| Experiment comparison | Completed local runs discovered from real config/result files; compare equal protocols |
| Plain-language commands | Local grammar; ball boop and behind-ball missions passed single deterministic physics trials; browser contact also verified |
| Approach | Nearby static-target trial passed; arbitrary navigation not solved |
| Follow and push | Experimental controller; failed recorded trials; no general success claim |
| Stop/recover | Analytic standing pose and explicit simulation reset; no learned self-righting |
| Jev decision model | License/inspiration audited; no model weights, training, or inference integrated |
| MJX backend | CPU and RTX 5090 probes passed, including forced timeout/reset; writable-array regression tested |
| Runpod | CLI checksum and read-only authentication verified; live catalog inspected |
| RTX 5090 GPU training | One completed experiment: 1,048,576 steps in 121.7 s; native CPU recheck 0.337 m/s, no falls in five trials. One failed interface attempt retained. Both pods deleted. |

The research server has no user accounts and binds to loopback by default. Host
it behind private port-forwarding. It is not a public production service. The
hardware model is original primitive geometry, not a validated real robot.

Bundled, checksum-verified CPU and GPU-trained checkpoints are seeded by setup.
The GPU-trained policy runs in the local CPU arena; participants do not need a
GPU just to play. Following, pushing-to-goal and learned recovery are still unsolved.
Windows/Apple Silicon installation and a full two-hour multi-user soak test were
not run here. Hosted CI evidence is recorded in `VALIDATION.md`;
for the source SHA and run link, separate from the local evidence.
