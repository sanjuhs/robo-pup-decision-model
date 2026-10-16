# Prototype status

Updated 2026-10-16. The native arena now has a **genuine pretrained Qwen3-VL
vision-language planner**, alongside the compact trained v4 planner and PPO motor.
A real model-generated fetch → hop → stop mission passed on CPU and an RTX 4090.
The GPU trial took **23.45 wall seconds for 19.98 simulated seconds**, with physics
deliberately paused during VLM inference; this is not a real-time browser result.

A separate **genuine guarded browser mission passed** on the CPU backend:
fetch → hop → stop in **403.790 mission wall seconds**, with physics continuing
in a standing pose during the model waits. Three real Qwen calls and independent
contact/lift/release/landing checks are retained with the
[full actual recording](../benchmarks/vlm-2026-10-16/browser-guarded/actual-run.webm).
This confirms one complete browser workflow, with long CPU waits; it does not
establish interactive GPU latency or broad VLM task reliability.

**Reliable visual planning remains incomplete:** a conditional command produced
the same look/fetch/stop plan for real, blank and turned images. The blank-image
response violated the visibility condition, despite valid JSON.

The native executor now adds a separate **eye-visibility gate** using 821
unchanged trained v4 pixel weights on a fresh activation image after generation.
The [small developmental probe](../benchmarks/vlm-2026-10-16/eye-grounding/probe.json)
observed no false positives/negatives over 18 positive and 50 negative pairs from
17 same-color, same-renderer views. It blocks ungrounded target actions visibly;
it does not repair Qwen's failed raw visual reasoning or establish general vision.

The [CPU mission and twelve checks](../benchmarks/vlm-2026-10-16/cpu-verified-mission/result.json),
[actual GPU mission and failed visual probes](../benchmarks/vlm-2026-10-16/gpu-4090),
[model guide](VLM_GUIDE.md), [validation](VALIDATION.md), [benchmarks](BENCHMARKS.md)
and [source-audited RoboJev comparison](ROBOJEV_COMPARISON.md) distinguish the models
and the evidence.

| Capability | Current evidence |
| --- | --- |
| Large vision-language model | Qwen3-VL-2B-Instruct: 2,127,532,032 parameters; 4,266,647,442 downloaded bytes; pinned Apache-2.0 public checkpoint. Pretrained externally; no RoboPup fine-tuning |
| Robot-eye model input | Actual 384×384 MuJoCo head-camera RGB + text + virtual energy/held state. Three real calls produced each successful CPU/GPU fetch/hop/stop mission; conditional visual grounding failed on blank imagery |
| Sequential execution | Strict supported-step JSON, physical checks, remaining-task memory and fresh-eye target-activation guard. Budgets: 360 execution-simulation seconds excluding model wait, 30 minutes wall time; earlier genuine failures retained |
| Eye-activation gate | 821 frozen v4 pixel weights on actual current eye RGB; ball/crate/beacon visibility required for respective target actions. Held-ball return and untargeted skills allowed; visible rejection, no automatic replacement plan |
| Native control boundary | VLM sees the robot eyes; existing v4 low-level skills retain overhead RGB. PPO and analytic controllers execute motion; not an end-to-end egocentric learned policy |
| CPU MuJoCo and residual PPO | Tested headless simulation and fresh-weight participant training; v4 planner and two motor demos, with archived v2/rejected v3 weights |
| Compact trained planner | v4: 246,712 parameters; 1,011,137-byte checkpoint. GPU-trained transformer plus CPU-refined task/needs decoder; six proposals, supervised arena imitation |
| Compact perception/language | Learned 96×96 overhead arena pixel classes and bounded English. Earlier fresh 6,000-sample v4 test: 100% intent / 99.967% first action; not a Qwen score or general visual/language benchmark |
| Approach / behind / fetch / push / follow | Compact v4 passed three fresh starts each. New real-Qwen CPU mission separately verified contact, 0.351 m held lift, return/release, then hop/landing and stop |
| Walk / run / hop | Analytic physical skills, not learned canine gaits. Real-Qwen CPU mission measured 0.26 s hop flight and four final foot contacts. Walking's brief measured flight remains a biomechanical limitation |
| Energy | Virtual play-energy, rest choice and executor recharge hysteresis; no battery or biological claim |
| Voice | Offline English PocketSphinx command grammar; synthetic fixture tested. Transcript requires Send; human accent/noise accuracy unmeasured |
| Browser presentation | Original procedural research robot with head optics and the existing contact-gated magnetic lifting tool; alternate golden-puppy skin remains selectable. Geometry follows physical poses |
| Appearance boundary | No photoreal dog, downloaded anatomical asset or independently simulated six-axis picking arm is claimed |
| Application checks | Latest full Python suite: **226 passed in 23.60 s**, with explicit dev/MJX/VLM extras; Ruff passed. Isolated browser suite: **21/21 in 117.847 s**; initial timeout retained |
| Genuine VLM browser | Guarded actual-Qwen mission passed on CPU with no model-response fixture or reset; **403.790 s** mission wall, **0.3621 m** held lift, fetch release **0.3901 m** from home, **0.28 s** hop flight and four-foot landing |
| Participants | Independent browser worlds and fresh-policy experiments. Earlier five-browser isolation and short compact-v4 capacity probes passed; no shared world or VLM multi-user capacity measurement |
| Local model serving | Private `/v1/chat/completions` route serves the configured local VLM; mandatory workshop key, bounded inline PNG/text requests and two pending/running serving slots. Controlled-adapter HTTP tests cover its contract |
| Local Vercel model chain | Real `vercel dev -L` gateway → injected private planner binding → native Qwen passed a camera/text stop-plan request in **42.4086 s**, six checks true; planning only, no physics. Seven helper ports closed afterward |
| Deployment | Three Vercel services prepared. Non-root 1.41 GB CPU container passed EGL, HTTP and authenticated advancing WebSocket smoke checks; cloud publication remains separate |
| Hosted Vercel | Not published. Personal-account authentication is unavailable and the requested service/route review remains outstanding; local routing is not a deployed service |
| GPU evidence | Actual RTX 4090 VLM inference and physical mission passed; seven static outputs valid JSON, visual condition failed. Two RTX 5090 allocations failed; earlier RTX 5090 motor/planner training remains recorded |
| Spending / cleanup | New RTX 4090 estimate $0.08326; cumulative confirmed-experiment estimate $0.79004 of $30. Latest pod deleted/absence verified; all six confirmed experiment pods deleted. Final invoices/storage reconciliation pending |
| Weights | Tiny trained policies/planners in GitHub `demo/`; large pretrained Qwen snapshot downloaded from Hugging Face into ignored `artifacts/vlm/`, with public provenance manifest. No new Hugging Face model/Space published |
| Jev | Source/license audited. RoboJev's public runner consumes sensory text and returns answer probabilities; no Jev checkpoint or calibrated learned judge integrated |
| Recovery | Standing/stop and explicit world reset; no learned self-righting |
| Browser-only/offline phone | Not implemented. Three.js runs on device; native MuJoCo and model inference require a reachable Python backend |

The earlier `planner.py::MultimodalPlanner` remains an untrained 769,149-parameter
research scaffold. The small live model is `ArenaPlannerV4`; the optional large
model is an unchanged pretrained Qwen checkpoint. Their checkpoints and training
claims are separate.

The latest full local suite passed **226 Python tests in 23.60 seconds**, with
explicit dev/MJX/VLM extras; the browser/render suite passed **21 checks**. The first browser-suite run had twenty passes and a puppy-control
30-second timeout; the unchanged targeted and isolated full reruns passed, without
altering assertions, physical thresholds or test timeouts. That fixture-based
suite is separate from the passing genuine-Qwen browser mission. Its first actual
capture failed a post-hop final-ball harness check and remains retained. The
corrected capture independently records fetch completion before the later hop;
[its result and original-time video](../benchmarks/vlm-2026-10-16/browser-guarded)
passed without a response fixture or world reset.

Hosted checks passed for guard source `2c0eeffa58ccf35fde7ae0ca00aecbd8a0f684f5`
([run 38921051265](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38921051265)).
The final runtime/frontend source `8ef5721cbdefe36f7f575ca0d721d9d74f66fdc0`
also passed [run 38921897635](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38921897635).
The latest [full local 226-test record](../benchmarks/vlm-2026-10-16/final-local-validation.json)
includes the credential-scanner changes. Its source
`77bf40c1f0819cab5175621d3090d2991b850eae` also **passed**
[run 38923264485](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38923264485),
with the [raw conclusion](../benchmarks/vlm-2026-10-16/final-published-source-ci.json)
retained. Exact source-specific scope and failures are in [VALIDATION.md](VALIDATION.md).

Not validated: real hardware, sim-to-real, arbitrary objects/language/terrain,
long-horizon obstacle avoidance, reliable image-conditioned decision quality,
photoreal assets, real mobile FPS, human voice accuracy, Windows/Apple Silicon
installation, long-duration concurrent VLM planning or a full two-hour workshop
soak. The private workshop server has a 12-session limit and optional workshop
access-key/origin controls; it is not a multi-tenant public robotics service.
