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
3. Add a MuJoCo Playground MJX/Warp path using a supported quadruped environment;
   benchmark against the native baseline on approved GPU hardware.
4. Define a policy transfer boundary: Playground policies cannot be loaded into
   this custom model without matching robot, observations, actions, and normalization.
5. Add an optional Jev-compatible physical-judgment adapter after collecting
   counterfactual outcomes. Keep network/model calls out of the motor-control loop;
   measure calibration, latency, and budget independently. No Jev weights are downloaded now.
