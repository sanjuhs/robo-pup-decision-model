# What is the model?

Measured 2026-10-16. Reproduce with
`uv run python scripts/inspect_models.py --out artifacts/model-inventory.json`.
The committed [current inventory](../benchmarks/gpu-planner-v4/model-inventory.json) includes checkpoint
hashes and exact counts; the command does not download or train anything.

## Generative robot-eye vision-language model

The optional high-level model is **Qwen/Qwen3-VL-2B-Instruct**, an Apache-2.0
pretrained foundation model with **2,127,532,032 parameters**. The pinned public
revision is `89644892e4d85e24eaac8bacfd4f463576704203`; its downloaded files total
4,266,647,442 bytes. The model manifest and first actual CPU image-inference
record are in [`benchmarks/vlm-2026-10-16/`](../benchmarks/vlm-2026-10-16).

This model receives actual perspective robot-eye RGB plus text/energy/held state
and verified action history. It generates a bounded JSON mission; the executor
checks physical outcomes and requests a fresh-eye remaining plan between steps.
It was not trained or fine-tuned in this project. Its local weights live under
ignored `artifacts/vlm/`; no token is needed for the public download. CPU latency
in the first probe was 127.73 seconds including load, overlapping other checks.
Subsequent actual closed-loop fetch → hop → stop trials passed on CPU in 183.27
wall seconds and on a provider-reported RTX 4090 in 23.45 seconds. Each used three
fresh-eye model calls and paused physics during inference. The GPU checkpoint
used 4,471,898,112 bytes peak PyTorch CUDA allocation in that trial.

The frozen static probes exposed a visual-reasoning failure: real, blank and
turned-away images produced identical look/fetch/stop plans. All seven outputs
were valid JSON, but this does not establish semantic correctness or visual
grounding. An earlier CPU run also repeated already completed fetches; its raw
failure motivated the explicit remaining-task memory guard. These limits and
all raw trials are retained in the [VLM evidence](../benchmarks/vlm-2026-10-16/).

The native executor separately reuses the unchanged **821-parameter** arena pixel
classifier on fresh perspective RGB before activating a targeted action. It
abstains when a sufficiently confident connected target region is absent. This
engineering guard uses no simulator object coordinates or segmentation at
inference, and its thresholds are not probability calibration. The small
[view/occlusion probe](../benchmarks/vlm-2026-10-16/eye-grounding/) is limited to
the same colored arena objects and renderer; it is not evidence of general VLM
vision improvement. No new training was performed for this guard.

See [the VLM guide](VLM_GUIDE.md), [RoboJev comparison](ROBOJEV_COMPARISON.md) and
[deployment services](DEPLOYMENT.md) for the difference between generative
planning, the small learned controller below, analytic mechanics and hosting.

## Trained arena transformer

The default live planner is `ArenaPlannerV4`: **246,712 parameters**, **1,011,137 bytes**
in `demo/arena-planner/planner.pt`, with digest, vocabulary and release metadata.
It has a learned arena pixel classifier, visual/text/state encoders, two width-96
four-head transformer layers, six plan queries and a 6,739-parameter neural task/needs
decoder. The transformer predicts intent; the decoder receives that predicted
intent, held state, normalized virtual energy and slot index. Its skill/object
logits are invariant to motor/yaw changes when the predicted intent/needs stay fixed.
RGB perception grounds the selected object targets. Inputs are actual overhead
96×96 RGB, up to 96 word tokens, and 50 motor observations plus virtual play-energy,
held state and yaw/pi. Outputs are skill/object logits and relative offsets; each
row is displayed as a fixed one-second proposal. Replanning occurs every half
simulated second, with persistent goal execution and physical completion checks.

It was trained by supervised imitation on generated arena labels. It is not a
general VLM, language generator or RL-trained task planner. On a new independent
500-scene / 12-template test (6,000 scene/command/state combinations), v4 measured
100% intent, 99.967% first action, 99.975% horizon skill and 99.95% object accuracy.
The v2 baseline on those same samples scored 91.667% intent and 59.817% first action.
Both passed 24/24 fresh canonical-command physical trials. Known older occlusion
failures remain visible; broader vision/language and navigation remain unproven.

v4 inherits v2/v3 weights, uses 101.13 s of RTX 5090 optimization and 87.26 s of
CPU-only decoder refinement. The pixel classifier stayed frozen. v3 was rejected
after a held-state application failure; a v4 bootstrap failed on network access.
Neither failed record is hidden. v2/v3 checkpoints are retained alongside the new
default, with lineage and SHA-256. See [experiment protocol/results](GPU_PLANNER_EXPERIMENT.md)
and [training/limitations](MULTIMODAL.md). Voice remains separate offline
PocketSphinx English recognition, with an approximately 28 MiB installed speech
model. No external model API or Hugging Face upload is involved.

## Trained motor controller

Both bundled policies use the same small Stable-Baselines3 actor/critic MLP.
PPO is the training algorithm; it does not imply a transformer architecture.

| Item | CPU curious-v2 | RTX 5090 policy |
| --- | ---: | ---: |
| Checkpoint path | `demo/curious-v2/policy.zip` | `demo/5090-once/policy.zip` |
| Archive bytes | 224,113 (219 KiB) | 259,942 (254 KiB) |
| Total policy parameters | 15,705 | 15,705 |
| Raw float32 policy parameter bytes | 62,820 | 62,820 |
| Deterministic actor parameters | 8,204 | 8,204 |
| Raw deterministic actor parameter bytes | 32,816 | 32,816 |

The actor is `50 → 64 → 64 → 12`, with Tanh hidden activations. A separate
`50 → 64 → 64 → 1` critic estimates value during training; 12 learned log-standard
deviations specify the training action distribution. Archive size includes
training/optimizer state and metadata, so different archive sizes do not imply
different network architectures. The raw actor size excludes normalization/runtime
code, the analytic gait, physics, browser, activations, and Python/PyTorch.

The 50 inputs are six base velocities, three orientation/gravity values, 12
joint positions, 12 joint velocities, two commands, sine/cosine gait phase,
body height, and 12 previous actions. The 12 outputs are bounded joint-target
residuals scaled by 0.3 radians and added to an analytic diagonal trot.
The policy acts at 50 Hz; physics steps at 200 Hz. It has no RGB, microphone,
language encoder, transformer, object detector, memory, or learned fetch skill.

The default neural hierarchy uses image-derived object targets. Its executor reads
own pose/proprioception and contact/true-state outcome checks. The selectable
historical rules controller uses privileged object positions and a local command
grammar; compare it explicitly as an oracle-style baseline. The motor reward's
energy term is squared torque, distinct from the new virtual play-energy variable.
Stop uses analytic stance and reset restores the world.

## Where are the weights?

Current and historical checkpoints are committed to this GitHub repository with SHA-256 sidecars.
Setup verifies and copies the two motor checkpoints into `artifacts/baseline/curious-v2/` and
`artifacts/runpod/5090-once/`. New participant runs save their own `policy.zip`
and intermediate checkpoints under `artifacts/<participant>/<run>/`.

Nothing has been published to Hugging Face. A Hugging Face **model repository**
would store weights; a **Space** would host an app. Neither is required here.
No external model service is called by the live arena. SB3/PyTorch checkpoints
must only be loaded from trusted authors; their deserialization is not a sandbox.

## Was the RTX 5090 actually used?

Yes for the new transformer: `benchmarks/gpu-planner-v4/attempt2/` contains actual
hardware, PyTorch 2.9.0+cu129/CUDA 12.9, capability 12.0, GPU utilization and
101.13-second training evidence. Rendering and native physics need no GPU in the
local demo. The small final decoder was subsequently refined on CPU.

The following paragraph describes the earlier motor experiment:

Yes. The recorded [hardware probe](../benchmarks/runpod-attempt-2/hardware.csv)
reports NVIDIA GeForce RTX 5090 with 32,607 MiB. The recorded JAX device is
`cuda:0`. MJX batched physics ran on GPU, while the small PPO optimizer ran on CPU.
The completed experiment took 121.7 learning seconds for 1,048,576 transitions;
83.1 seconds of JAX compilation and provisioning time are separate.

One interface attempt failed before the repaired retry. Both pods were deleted.
Their aggregate estimated GPU charge is $0.2769 at the recorded $1.19/hour rate;
this is not a final billing receipt. It did not train a 2B vision-language model.
See [the complete evidence and limitations](BENCHMARKS.md).

## Three-dimensional character versus neural model

The golden puppy is original procedural Three.js artwork: 18,248 triangles,
382,736 bytes of unique geometry buffers, 18 physical attachments, and a generated
256×256 fur-bump texture. There is no downloaded GLB or texture asset. The whole
browser JS build, including Three.js, is about 552 kB / 143 kB gzip. Those values
are separate from the neural checkpoint and do not include Python dependencies.

The look is stylized, not photoreal. Its simple articulated segments inherit the
robot's leg poses, not a detailed anatomical canine skeleton. A production canine
asset would need an audited license, a rig, physically consistent retargeting,
better fur/skin shading, and measured mobile performance. A 5–15 MB compressed
character is a possible delivery target, not an existing file or benchmark.

## Can it run entirely on a phone or in an offline tab?

The browser renderer runs on the device, but MuJoCo and policy inference currently
run in a local Python server. A small policy does not make the entire stack small:
PyTorch and other installed runtime packages are much larger than its weights.
A phone can display a reachable private backend; a fully offline, browser-only
simulation/inference bundle has not been built or tested.

That path would require browser physics/WASM plus a deterministic actor export
(for example ONNX), the exact gait/observation interface, numerical parity tests,
and real device benchmarks. Merely exporting the 32 KiB actor is insufficient.

## Historical transformer scaffold

`src/robopup/planner.py::MultimodalPlanner` has 769,149 parameters and remains
untrained and disconnected. It has different image-history/byte-token/output
interfaces from the deployed `scene_planner.py::ScenePlanner`. Its tensor tests
are research infrastructure, not evidence for learned planning. No random
scaffold weights are shipped. See [the live implementation](MULTIMODAL.md).
