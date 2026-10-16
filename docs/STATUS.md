# Prototype status

Updated 2026-10-16. See [benchmarks](BENCHMARKS.md), [model card](MODEL_CARD.md)
and [multimodal implementation](MULTIMODAL.md) for evidence and boundaries.

| Capability | Current evidence |
| --- | --- |
| CPU MuJoCo and residual PPO | Working headless simulation and fresh-weight participant training; three bundled trained checkpoints across motor/planner layers |
| Trained multimodal planner | 232,414 parameters; RGB + English tokens + numeric state; six action/target proposals, live execution and feedback |
| Perception | Learned arena pixel classes; dedicated 96×96 overhead RGB. Final-test centroid error 7.21 cm vs segmentation centroids; not general vision |
| Language | Finite task vocabulary and learned phrase compositions; final test 97.875% intent / 89.542% first action, weak unseen fetch wording |
| Approach / behind / fetch / push / follow | Three varied-start RGB physics trials each passed. Fetch uses an explicitly magnetic lifting tool with contact, return and release checks. General navigation remains unproven |
| Walk / run / hop | Three trials each passed movement/flight criteria. Analytic physical skills, not learned canine gaits; walking has brief measured flight and is not biomechanically validated |
| Energy | Virtual play-energy, visible UI, learned rest choice and executor recharge hysteresis; no battery or biological claim |
| Voice | Offline English PocketSphinx command grammar; synthetic fixture tested. Transcript requires Send; human accent/noise accuracy unmeasured |
| Browser | Articulated original golden puppy, camera and plan inspection, orbit/follow camera, manual controls, pause/reset, experiments and policy switching |
| Appearance | Stylized procedural skin attached to physics; not photoreal or anatomical canine rig |
| Participants | Separate browser worlds and fresh-policy experiments; five-browser isolation tested; short active-neural probes: five at ≥0.819×, ten at ≥0.571× real time. No shared world |
| RTX 5090 | Prior completed experiment: 1,048,576 steps in 121.7 s; native replay 0.337 m/s, 0/5 falls. Failed interface attempt retained, both pods deleted |
| Spending | Prior estimated GPU charge $0.2769, not final invoice. Multimodal training/evaluation ran on this CPU host with $0 additional Runpod usage |
| Weights | GitHub `demo/`; checksum sidecars. No Hugging Face model repository or Space |
| Jev | Inspiration/license audited. No Jev weights, adapter or reproduced Jev benchmark |
| Recovery | Standing/stop and explicit world reset; no learned self-righting |
| Browser-only/offline phone | Not built; Three.js runs on device, physics/inference need the local Python server |

The earlier `MultimodalPlanner` class is still an untrained research scaffold.
The trained arena uses the distinct `ScenePlanner` class and bundled checkpoint.
Historical failed follow/push/retrieval and language runs remain in `benchmarks/`.

Not validated: real hardware, sim-to-real, arbitrary objects/language/terrain,
long-horizon obstacle avoidance, photoreal assets, real mobile FPS, human voice
accuracy, Windows/Apple Silicon installation, long-duration concurrent neural planning or a
full two-hour workshop soak. The server is an unauthenticated private workshop
tool with a 12-session limit and same-origin checks, not a public hosted service.
