# Architecture

```text
Browser / Three.js                   Local experiment process
  orbit + follow camera                JSON experiment config
  geometry + live telemetry            fresh SB3 PPO MLP (64 x 64)
  behavior controls                    4 native MuJoCo environments
          | WebSocket                         |
          v                                   v
FastAPI / one process                CSV + Monitor episode logs
  one MuJoCo world per connection     checkpoints + result.json
  rules -> speed/yaw command                  |
  analytic trot + optional PPO         Experiments API + policy menu
  position actuators -> physics
```

## Robot and environment

Original 12-actuator primitive dog: roll, pitch, and knee on each of four legs.
Position actuators apply bounded torques; there is a floating base, gravity,
friction, contacts, two static obstacles in the yard, and a free crate.
Physics steps at 200 Hz; actions at 50 Hz; the browser receives transforms at a
target 25 Hz. Browser frames interpolate actual simulated geometry.

The browser has two visual skins on the same physical transforms. Each scene
geometry includes its MuJoCo body name so the original golden-puppy artwork can
attach to the correct torso/limb without relying on numeric geometry IDs. Its
18 attachments add soft ellipsoids, paws, muzzle and coat materials; eyes, ears
and tail have cosmetic expression. Static details are batched by material. No
additional limb cycle, torso bobbing, root motion, or physical contact is scripted.
The collision model and policy interface remain the original primitive robot.

Observations: base velocity, orientation/gravity row, joint positions/velocities,
command, gait phase, height, and previous action (50 values). Actions: 12 residual
joint targets, clipped to [-1,1], scaled by 0.3 radians, then clipped to joint limits.
The gait prior uses two-link inverse kinematics and a diagonal trot.

Reward: forward-speed tracking, upright posture, squared-torque cost, residual
smoothness, and a fall penalty. Episodes end on a fall or after 12 seconds.
Training randomizes initial joint angles and floor friction. The metric called
energy is a squared-torque proxy, **not** joules or electrical power. The reported
speed is body-frame forward velocity; world-X displacement is separately recorded.
Heading and lateral drift are not penalized in this first reward.

## Decisions

The deterministic hierarchy selects approach, follow, push, stop, manual, or
recover. A fallen robot enters recover and requests a posture reset. There is no
trained self-righting controller. Stop uses an analytic standing pose, even when
a learned policy is selected. The current PPO policies were trained only for
forward flat-ground commands, so turning and other commands are distribution shifts.

The local command grammar accepts ball approach, behind-ball waypoints, follow,
push, stop, and recovery requests. It is not an LLM. Ball-boop completion requires
actual robot/ball contact; behind-ball completion requires the robot to be on the
ball's +X side by more than 0.35 m, within 0.45 m on Y, after consuming waypoints.
The ball has rolling friction and low physical perimeter walls contain it.

The moving follow beacon travels at 0.072 m/s. Push aligns behind a crate and
commands contact toward a goal at (3,0); it is a heuristic, not a learned manipulation
skill. Behavior trials record both successes and failures.

## Isolation and trust

Each browser connection owns its simulation. The process accepts at most 12
sessions and rejects cross-origin sockets. A 0.5-second stale-manual-input timeout
returns commands to zero. Training is a CLI operation, not a public HTTP endpoint.
Experiment IDs cannot traverse directories; policy loading is limited to completed
local runs inside the artifact root. SB3 checkpoints should only come from trusted
authors, as PyTorch/cloudpickle deserialization is not a sandbox.

This is a private workshop tool, not a hosted multi-tenant service. Participant
files share a local filesystem; there are no accounts, quotas, persistent browser
worlds, TLS termination, or collaborative shared-world synchronization.

## Next research milestones

1. Add commanded yaw/velocity curricula and long-horizon drift penalties.
2. Train separate approach/push/recovery skills with task-specific held-out tests.
3. Extend the optional custom MJX backend, already probed on RTX 5090, with domain
   randomization parity and a standard Playground benchmark environment.
4. Define a policy transfer boundary: Playground policies cannot be loaded into
   this custom model without matching robot, observations, actions, and normalization.
5. Add an optional Jev-compatible physical-judgment adapter after collecting
   counterfactual outcomes. Keep network/model calls out of the motor-control loop;
   measure calibration, latency, and budget independently. No Jev weights are downloaded now.

## Multimodal planner research boundary

The trained 15,705-parameter motor MLP is separate from the new
`MultimodalPlanner` scaffold. The latter has 769,149 random-initialized parameters,
RGB-patch/history, byte-text and state encoders, a transformer, and horizon heads.
Only tensor contracts/gradients are tested. It is not imported by the arena
server, has no trained checkpoint and must not be used for live control.
See [the model card](MODEL_CARD.md) for exact sizes and
[the multimodal design](MULTIMODAL.md) for closed-loop execution, data collection,
calibration, unavailable-skill masking and perception evaluation requirements.

## Optional GPU physics

`MJXVecEnv` batches the same primitive robot and 50-value observation / 12-value
action interface on the JAX device. `train_mjx` refuses a CPU device when
`--require-gpu` is set. It records compile time separately and checks a forced
timeout/reset before learning. Writable copies at the SB3 boundary are necessary:
PPO modifies timeout rewards in-place. The first GPU attempt exposed this defect;
it is regression-tested and retained in the evidence.

GPU training uses 128 worlds, 64-step rollouts, batch size 1024, fixed friction 0.9,
and a CG solver with 10 iterations / 4 line-search iterations. Native CPU training
uses four worlds, 256-step rollouts, batch size 256, randomized friction, and the
native default solver. Their raw throughputs are useful operational measurements,
not a controlled hardware speedup comparison. PPO optimization remains on CPU;
the physics is GPU accelerated. The saved SB3 checkpoint can be evaluated on the
native environment without CUDA or JAX.
