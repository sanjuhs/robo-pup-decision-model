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
  RGB/text/state transformer + executor                  |
  PPO trot + analytic physical skills         Experiments API + policy menu
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
The leg collision model and 50-input/12-action motor interface remain the original
primitive robot. The interactive PetEnv adds a thirteenth gripper slide actuator,
a contact-gated weld, visual home/beacon markers and an overhead RGB camera.
The default training environment excludes those additions for replay compatibility.

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

## Decisions and physical skills

The default `BrainSession` captures actual 96×96 RGB every 0.5 simulated seconds.
The trained `ScenePlanner` fuses learned pixel/centroid tokens, word tokens and
53 numeric inputs. Six query rows predict skills, object choice and metric offsets.
Chunk durations are fixed at one second; softmax scores are not calibrated.
`GoalExecutor` persists the instruction, grounds targets in the robot's own pose,
steers, checks contact/landing/return outcomes, and holds or stops on uncertainty,
invisible targets, unsupported instructions, falls or timeout. Later proposal rows
are informational; each immediate step is reconsidered from fresh observations.

True object coordinates are not navigation inputs in this mode. They verify task
outcomes separately. Proprioception includes own pose/yaw, and home `(0,0)` and
crate goal `(3,0)` are task constants. The camera is north-up overhead, not the
Three.js viewer or an egocentric eye. Training segmentation/oracle labels are
isolated in `planner_data.py`; the explicitly labelled oracle mechanics evaluator
is never imported by the live arena.

Fetch uses physical tool/ball contact to activate a magnetic weld at the existing
relative pose. A slide servo lifts the tool. The executor returns using a visual
home target, settles, lowers, releases and checks the ball on the floor within
0.6 m of home, with prior contact/hold/lift. No object qpos edits occur during a
grasp. Explicit reset and offline scene construction do set initial poses.

The default curious-v2 PPO residual is used only in near-straight trot; steering,
four-phase walk, faster diagonal run, crouch/extend hop and gripper control are
analytic physical controllers. Hop temporarily raises actuator gain/torque bounds
and restores them afterward; no external force or visual root animation creates
flight. Run has measured flight; the four-phase walk still has brief aerial
intervals and is not a validated canine walk. Virtual play-energy drains with
activity, recharges at rest, and can cause the planner to rest before resuming.

The selectable historical rules baseline uses simulator object positions and a
local text grammar. It retains its earlier failures; it is not evidence for vision.
The moving beacon travels at 0.072 m/s. No learned self-righting is implemented.

Voice uses browser AudioWorklet capture, local WAV transfer, offline PocketSphinx
English command grammar, then explicit transcript Send. Same-origin audio checks,
300 kB upload bounds, 0.1–8 second WAV bounds and two concurrent decodes constrain
this private endpoint. Audio is not persisted. Speech recognition is separate
from the trained RGB/text/state fusion model.

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

## Distinct model boundaries

The trained 15,705-parameter PPO motor MLP, trained 232,414-parameter `ScenePlanner`
and untrained 769,149-parameter historical `MultimodalPlanner` are distinct models.
Only the first two have bundled trusted weights. The historical scaffold is not
imported by the arena server. See [model card](MODEL_CARD.md) and
[training and evaluation](MULTIMODAL.md) for data, sizes and measured limitations.

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
