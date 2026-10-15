# Prototype status

This file separates implemented paths from measured task success. See
`BENCHMARKS.md` and the raw JSON in `benchmarks/` for evidence.

| Capability | Current evidence |
| --- | --- |
| Native headless MuJoCo physics | Tested; physical quadruped, contacts, crate, obstacles and ball |
| PPO from fresh weights | Completed CPU runs; residual gait policies, not end-to-end torque learning |
| First locomotion policy | Curious-v2: 0.340 m/s body-frame forward speed on five 12-second held-out trials; no falls |
| Interactive browser | Tested desktop/mobile; live geometry, orbit/follow camera, keyboard, pause/reset, policy loading |
| Multiple participants | Five simultaneous independent browser worlds tested; no shared-world collaboration |
| Experiment comparison | Completed local runs discovered from real config/result files; compare equal protocols |
| Plain-language commands | Local grammar and physical ball; parsing/control routing tested; task success evaluated separately |
| Approach | Nearby static-target trial passed; arbitrary navigation not solved |
| Follow and push | Experimental controller; failed recorded trials; no general success claim |
| Stop/recover | Analytic standing pose and explicit simulation reset; no learned self-righting |
| Jev decision model | License/inspiration audited; no model weights, training, or inference integrated |
| MJX backend | Optional batched adapter added; CPU compiled finite-state probe passed; GPU validation pending |
| Runpod | CLI checksum and read-only authentication verified; live catalog inspected |
| RTX 5090 GPU training | One authorized run being prepared; no success claim before results |

The research server has no user accounts and binds to loopback by default. Host
it behind private port-forwarding. It is not a public production service. The
hardware model is original primitive geometry, not a validated real robot.
