# Prototype status

Updated 2026-10-16. See [benchmarks](BENCHMARKS.md), [model card](MODEL_CARD.md)
and [multimodal implementation](MULTIMODAL.md) for evidence and boundaries.

| Capability | Current evidence |
| --- | --- |
| CPU MuJoCo and residual PPO | Working headless simulation and fresh-weight participant training; current v4 planner and two motor demos, with archived v2/rejected v3 weights |
| Trained multimodal planner | v4: 246,712 parameters; GPU-trained transformer plus CPU-refined neural task/needs decoder; RGB + English + numeric state; six proposals and physical execution |
| Perception | Learned arena pixel classes; dedicated 96×96 overhead RGB. Final-test centroid error 7.10 cm vs segmentation centroids; not general vision |
| Language | Bounded vocabulary/compositions; fresh 6,000-sample test: 100% intent / 99.967% first action. Frozen test, not general-language competence |
| Approach / behind / fetch / push / follow | Current v4 passed three fresh starts each; fetch 18.34–19.52 simulated seconds with physical contact/lift/return/release. Older occlusion failures remain retained; general navigation unproven |
| Walk / run / hop | Three trials each passed movement/flight criteria. Analytic physical skills, not learned canine gaits; walking has brief measured flight and is not biomechanically validated |
| Energy | Virtual play-energy, visible UI, learned rest choice and executor recharge hysteresis; no battery or biological claim |
| Voice | Offline English PocketSphinx command grammar; synthetic fixture tested. Transcript requires Send; human accent/noise accuracy unmeasured |
| Browser | Articulated original golden puppy, camera and plan inspection, orbit/follow camera, manual controls, pause/reset, experiments and policy switching |
| Appearance | Stylized procedural skin attached to physics; not photoreal or anatomical canine rig |
| Participants | Separate browser worlds and fresh-policy experiments; five-browser isolation tested; short active-neural probes: v4 minimum ratios about 0.814× for five and 0.570× for ten real time. No shared world |
| RTX 5090 | Motor experiment: 1,048,576 steps in 121.7 s. Planner v3/v4: 8,000 CUDA updates in 95.49/101.13 s; v3 rejected by application test, v4 selected after CPU decoder stage |
| Spending | Follow-up GPU estimate $0.4298; aggregate session $0.7068, within $30. Five own pods deleted/absence verified. Final invoice/storage reconciliation pending |
| Weights | GitHub `demo/`; checksum sidecars. No Hugging Face model repository or Space |
| Jev | Inspiration/license audited. No Jev weights, adapter or reproduced Jev benchmark |
| Recovery | Standing/stop and explicit world reset; no learned self-righting |
| Browser-only/offline phone | Not built; Three.js runs on device, physics/inference need the local Python server |

The earlier `MultimodalPlanner` class is still an untrained research scaffold.
The trained arena uses `ArenaPlannerV4`, extending the original `ScenePlanner`,
and a bundled checksum-verified checkpoint.
Historical failed follow/push/retrieval and language runs remain in `benchmarks/`.

Not validated: real hardware, sim-to-real, arbitrary objects/language/terrain,
long-horizon obstacle avoidance, photoreal assets, real mobile FPS, human voice
accuracy, Windows/Apple Silicon installation, long-duration concurrent neural planning or a
full two-hour workshop soak. The server is an unauthenticated private workshop
tool with a 12-session limit and same-origin checks, not a public hosted service.
