# Validation record

Validated on the development Linux machine with Python 3.12.14, Node 24.19.0,
MuJoCo 3.3.7, PyTorch 2.9.0 CPU and optional JAX 0.7.2. Runpod hardware and source
hashes are recorded separately in `benchmarks/runpod-attempt-*/`.

## Current native activation-image guard

The [eye-grounding probe](../benchmarks/vlm-2026-10-16/eye-grounding/probe.json)
evaluates a new activation guard using **821 unchanged trained v4 pixel weights**.
After genuine model generation returns, the native mission recaptures head RGB
and checks the immediate target before enabling motion. The guard receives no
object coordinates, simulator IDs, segmentation or target truth during inference.

Seven recorded GPU images and seventeen fresh physical views are retained. For
the fresh views, offline renderer labels define visibility as at least 64 object
pixels: **18 positive and 50 negative object/view pairs**, with **zero observed
false positives or false negatives**. Those are 68 correlated pairs from a small
same-color, same-renderer developmental probe, not 68 independent scenes or a
general recognition benchmark. Thresholds were fixed before fresh rendering.

The engineering rules require posterior ≥0.9 and a sufficiently large connected
predicted region (64 pixels for the tested 384×384 images). They are not calibrated
success probabilities. The old overhead-trained arena color classifier has not
become a general visual encoder through this transfer. Missing evidence can mean
occlusion, distance, shadows or different colors.

Approach/behind/fetch require grounded ball pixels, push requires crate pixels,
and follow requires beacon pixels. Fetch may return a physically held ball;
look, turns, other untargeted skills and drop do not need target pixels. The
recorded blank/turned frames reject fetch. A missing target produces a visible
failure and holds movement; no alternate plan is silently invented, and Qwen's
failed original response remains unchanged.

Native mission budgets now allow **360 simulated execution seconds**, excluding
model-wait time, plus a **30-minute wall limit**. The earlier headless CPU/GPU
mission records below preceded this activation guard and are not retroactively
presented as a fresh guarded-model evaluation. The full new application suite
and corrected browser run require their own results.

The first genuine browser capture is retained as
[`browser-attempt1`](../benchmarks/vlm-2026-10-16/browser-attempt1). Its mission
reported completion, but the harness failed a final post-hop ball-position check
and had not saved the independent fetch-completion snapshot. Its video, original
harness and rejection remain available; it is not counted as a passing browser
mission. A corrected harness records the release/fetch transition before the
later hop, while retaining physical hop/landing checks.

## Actual RTX 4090 inference: successful sequence and failed visual semantics

The [GPU run records](../benchmarks/vlm-2026-10-16/gpu-4090) retain genuine
Qwen3-VL inputs/responses, CUDA runtime, native mission outcomes and deleted-pod
lifecycle. The model ran on an RTX 4090; the simulator remained native CPU.
The recorded source is `8ca421bd95e6241ea2f676f78a9af24449796354`.

- Seven static probes produced **seven schema-valid JSON missions**. This is a
  structured-output result, not a semantic or physical success rate.
- **The visual conditional test failed:** real, blank and turned-camera images
  produced identical look → fetch → stop sequences. The blank-image response
  violated the instruction to avoid fetching when no red ball is visible.
  A real attached image is confirmed; reliable image-conditioned decision making
  is not. The completed-history-only static prompt also repeated fetch.
- A separate corrected **fetch → jump → stop** native mission passed all twelve
  physical/trace checks with three genuine model calls. It took **23.4518 wall
  seconds / 19.98 simulated seconds**. Calls took 9.882, 5.567 and 5.405 seconds;
  the first includes 3.197 seconds loading. Physics was frozen during generation.
- One contact-gated grasp and a **0.35585 m** held lift were verified, followed by
  return/release, **0.28 s** hop flight and four-foot landing. The later hop moved
  the ball outside the home radius after fetch had completed. PPO residuals ran
  on 291 steps alongside analytic skills, without external forces.
- Provider hardware reported RTX 4090 and **49,140 MiB**, an atypical capacity.
  Torch peak CUDA allocation was 4,471,898,112 bytes. These are separate recorded
  quantities; neither is a standard-card capacity or production-memory guarantee.
- The pod was deleted and absence verified. GPU estimate **$0.08326**, cumulative
  confirmed-experiment estimate **$0.79004**, within $30; final invoice and storage
  reconciliation remain outstanding. Both preceding failed RTX 5090 allocations
  are preserved and are not counted as observed GPU runtime.

This is actual GPU model inference and a narrow model/controller success, while
visual semantic grounding remains an incomplete part of the user's goal. It is
not a real-time browser result, robust autonomous vision benchmark or new LLM
training result. The CPU trial and fixture-based application checks below remain
separate evidence.

## Genuine VLM, sequential native mission and deployment extension

The externally pretrained **Qwen/Qwen3-VL-2B-Instruct** checkpoint is pinned to
`89644892e4d85e24eaac8bacfd4f463576704203`, with **2,127,532,032 parameters**
and **4,266,647,442 downloaded bytes**. Its Apache-2.0 provenance and file hashes
are retained in [`model-manifest.json`](../benchmarks/vlm-2026-10-16/model-manifest.json)
and the [experiment protocol](../benchmarks/vlm-2026-10-16/protocol.json).
This is actual pretrained-model inference, not a new LLM training run or a rename
of the tiny v4 model.

The corrected native CPU trial requested **"Fetch the red ball, then hop, then
stop."** Three actual Qwen calls each received the corresponding saved 384×384
robot-eye image and produced validated mission JSON. The native engine verified
fetch → jump → stop in order; all twelve recorded checks passed. The full
[result](../benchmarks/vlm-2026-10-16/cpu-verified-mission/result.json), three
[inference records and PNGs](../benchmarks/vlm-2026-10-16/cpu-verified-mission)
retain the unedited responses, image hashes, timing, physical trace and outcomes.

- CPU BF16, four PyTorch threads; **183.2708 wall seconds**, **19.98 simulated
  seconds**, 999 physical control steps. Model calls took 62.233, 61.276 and
  56.315 seconds, including first-call load time.
- Contact/grasp occurred once, held-ball lift reached **0.35124 m**, and the ball
  was released on the floor within the home criterion. Hop flight was **0.26 s**,
  landing ended with four foot contacts and upright ratio 0.99999. The execution
  used 294 PPO-residual steps and no externally applied forces.
- **Physics was frozen while the VLM was thinking.** This headless CPU result does
  not establish real-time browser responsiveness. The physical controller still
  uses overhead-v4 grounding plus analytic skills; high-level robot-eye input
  alone is not an end-to-end egocentric control result.
- The first genuine trial repeated fetch during replanning and was rejected. Its
  [original records and rejection](../benchmarks/vlm-2026-10-16/cpu-rejected-replan)
  remain unchanged. The controller now supplies explicit remaining-goal memory
  and rejects plans that change the remembered remainder. That is an application
  prompt/validation fix; the Qwen weights were not optimized.

Before the new activation guard, the completed full local Python suite passed
**207 tests**; a monitor regression subsequently passed its targeted test. The
expanded guarded-application suite and current hosted CI require their own
recorded conclusions, not inference from those earlier runs.
Ruff, frozen dependency installation and browser build passed. All **21 browser/
render checks** passed, including the VLM UI using controlled provider fixtures,
original physics/real-v4 fetch, head-camera inspection, robot/puppy skin parity
and existing pause/camera assertions. These fixtures are labeled explicitly;
actual Qwen browser execution is not claimed by them.

The private local-model HTTP endpoint validates workshop authentication,
PNG/text/model/token bounds and local-only serving, with two pending/running
slots. Its HTTP tests use a controlled adapter and do not stand in for model
inference. Native mission contract tests separately perform real fetch/hop
physics with controlled high-level plans. Genuine Qwen evidence is the recorded
trial above, rather than either fixture suite.

Deployment checks are retained in
[`local-validation.json`](../benchmarks/vlm-2026-10-16/local-validation.json) and
[`container-smoke.json`](../benchmarks/vlm-2026-10-16/container-smoke.json):

- Actual Vercel CLI `vercel dev -L` served the Vite frontend and public gateway;
  the generated internal planner binding answered status requests. This was a
  local service run, not a Vercel cloud deployment or deployed private-network test.
- The non-root **1.41 GB CPU image** passed imports, actual 384×384 EGL eye
  rendering, HTTP frontend/health and an authenticated WebSocket advancing native
  physics from 0.04 to 0.08 seconds. Its disposable container was removed.
  The optional VLM-extra image was not built or tested.
- Failed Docker TLS/non-root-permission/disk-cache checks were diagnosed and
  corrected; retained summaries distinguish them from the passing smoke run.
  The three Vercel services are prepared, but personal-account authentication and
  the requested service/route review are still outstanding. Nothing was published.

Two new RTX 5090 allocation attempts failed (HTTP 500 and HTTP 400) without an
observed matching pod. The first attempt's
[reconciliation](../benchmarks/vlm-2026-10-16/gpu-allocation-attempt1/reconciliation.json)
and second [lifecycle](../benchmarks/vlm-2026-10-16/gpu-allocation-attempt2/lifecycle.json)
are retained. An allocation error or elapsed monitor time does not establish GPU
usage or billing. The later successful RTX 4090 run and its completed cleanup are recorded above;
its valid JSON is reported separately from failed visual semantics.

The following v4 and older sections retain their original version-specific counts,
measurements, cleanup state and hosted CI conclusions.

## Historical GPU-trained v4 release

- Local Python suite: **30 passed**. The original settled-pose held-state
  intervention is retained and now passes against v4, alongside physical fetch,
  RGB grounding/blank visibility, energy, speech and physics regressions.
- Full browser/render suite: **18 passed**. The camera/plan check explicitly asserts
  `arena-rgb-transformer-v4`; the recorded fetch captures actual v4 inference and
  measured contact/lift/return/release. Exact pause/idle-camera assertions remain.
- Frozen npm installation, browser build and Ruff passed. A separate HTTP/WebSocket
  smoke probe verifies the built page and advancing MuJoCo controls.
- Fresh final test: 500 mixed static/motion scenes × 12 new phrase compositions,
  6,000 task/state samples. Intent 100%, first action 99.967%, horizon 99.975%,
  objects 99.95%. Identical v2 baseline: intent 91.667%, first action 59.817%.
- Fresh physical protocol: both v2 and v4 **24/24**, seeds 707/808/909, unchanged
  outcome criteria. Fetch 18.34–19.52 s for v4 vs 74.44–77.36 s for v2; push slower.
- GPU v3 was rejected after its application regression despite good fresh-test
  scores. A v4 network/bootstrap failure is retained. The first CPU decoder-stage
  metrics are retained; final CPU decoder refinement used only grammar-grid
  validation before opening the new scene/language final test.
- One application test encountered an already queued streaming state while waiting
  for invalid-control rejection. It now requires the exact `Invalid controls`
  error within five messages; validation bounds are unchanged. Three independent
  diagnostic sockets also confirmed the actual rejection.
- All five session-created Runpod pods are absent, confirmed by CLI listing after
  deletion. GPU estimates total $0.7068; fresh attempts' final billing rows were
  not yet posted. This is not a reconciled invoice.

Raw measurements, hashes, source/initialization lineage, CUDA utilization, failures,
checks and cost ledger are in `benchmarks/gpu-planner-v4/` and the rejected run in
`benchmarks/gpu-planner-v3/`. The actual v4 video is
`docs/assets/gpu-planner-fetch.webm` (24.39 s, no speed change or synthetic motion).
Hosted CI **passed** for release source
`0a9beda987b98be6d4802888598bae966c6cd945`: [full prototype checks](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38878935098).
This includes fresh frozen setup, all 30 Python tests, lint and all 18 browser/
render tests. The public API step record is retained in
`benchmarks/gpu-planner-v4/hosted-ci.json`. Later evidence-only commits preserve
that tested runtime source.

The following sections retain their historical version-specific evidence.

## Multimodal extension

- Current Python regression run: 26 passed, including actual RGB counterfactuals,
  offline speech fixture/API boundaries, contact-driven lift/release, powered
  flight/landing with no external force and a complete trained RGB fetch trial.
- Ruff and browser build pass. The new build is approximately 552 kB JS / 143 kB
  gzip; Vite emits its bundle-size advisory.
- Independent 400-scene/12-template final test and all 24 physical trials are
  preserved in `benchmarks/multimodal-2026-10-16/`. See `BENCHMARKS.md` for exact
  metrics, failed first model, failed retrieval pilot and provenance limitations.
- Real offline PocketSphinx decoding passed on a locally generated Flite fixture.
  Fake-microphone browser capture/transcription passed; real-human speech accuracy
  remains unmeasured.
- Integration exposed a NumPy outcome boolean JSON error; conversion now preserves
  live socket output. A subsequent grasp-frame test exposed outcome telemetry one
  physics frame behind the held flag; post-step status now synchronizes contact
  outcomes. Assertions remain intact. An initial matrix test counted the header
  as an action; it now counts six body rows.
- Playwright's optional video binary download was blocked by its CDN policy.
  Actual-frame recording uses the existing system FFmpeg and CDP screencast instead;
  the functional browser tests do not require that optional recording binary.

The full local browser run passed **18/18**: the original 14 checks plus RGB
plan/camera/energy and real hop/landing, synthetic-microphone offline transcription,
a complete physical RGB fetch through the browser, and PCM capture/resampling.
Frozen setup was rerun successfully with all 26 Python tests. Active-neural
capacity probes passed advancement/isolation checks for five and ten sessions,
with minimum ratios 0.819× and 0.571× real time respectively; this is a short
capacity probe, not a completion or soak benchmark. Ruff and the final
browser build passed. The fetch recording preserves actual browser frames and
timestamps; it is not sped up. The earlier policy-menu display was stale while
telemetry already reported the correct CPU motor; the final UI now synchronizes
the selector and has an explicit parity assertion.

Hosted CI **passed** for final UI/controller source
`5a7f148a7b26118991f6542ea68e20ae1c08b2e9`:
[full prototype checks](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38870694044).
This includes frozen setup/Python tests, lint, Chromium setup and the complete
18-test browser/render suite. The controller-only revision `f80f4cb` also passed
[its hosted checks](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38870218910).
The public API conclusion/step record is retained in
`benchmarks/multimodal-2026-10-16/hosted-ci.json`. Later documentation-only commits
do not change the tested runtime source. The historical checks below retain
their original counts/scope.

## Previous release evidence

| Check | Observed result |
| --- | --- |
| `ROBOPUP_MJX=1 bash scripts/setup.sh` | Passed frozen Python/npm installation, two checkpoint digest checks, browser build, operator CLI verification and Python tests; repeat runs preserve local outputs |
| `pytest -q` | 19 passed, no skipped/disabled/expected failures; one upstream AnyIO deprecation warning |
| `ruff check src tests scripts` | Passed |
| `npm run build` | Passed; one approximately 547 kB JS bundle produces Vite's size advisory |
| `npm test` | 14 passed: nine browser checks including real geometry/controls, experiment results and policy loading, command grammar, physical CPU-policy ball contact, GPU-policy locomotion, close-up/follow camera and frozen expressions on pause, mobile layout, five simultaneous browsers, golden-puppy walking/appearance isolation; five deterministic render timing/reset/idle-camera checks |
| Post-optimization physical ball browser test | 1 passed; no contact assertions disabled |
| `scripts/smoke.py` against saved startup | Passed health, built HTML, advancing real MuJoCo state and a functional control request |
| `scripts/session_benchmark.py` | Final ten independent worlds: minimum 0.986 real-time ratio over 75 frames per session |
| Two bundled checkpoint SHA-256 sums | Verified; seeding into a fresh directory and rerunning preserves prior files |
| Runpod CLI/API | CLI 3.0.1 official checksum verified; actual read-only authentication succeeded |
| RTX 5090 experiment | One completed 1,048,576-step experiment, 121.7 s learning; one earlier failed interface attempt retained and fixed |
| Native replay of GPU-trained checkpoint | Five-seed summary matched the pod's native evaluation: 0.3368 m/s, 0/5 falls |
| Resource cleanup | Both created pod IDs deleted, absence confirmed, zero own pods remaining; no persistent volumes created |
| Billing | API calls succeeded but histories were empty at check time; aggregate GPU cost estimate $0.2769, not a final receipt |
| Source credential check | No Runpod API-key pattern in tracked text files; account key was not passed into the GPU job |
| Credential scanner regression | A synthetic key in a temporary Git repository fails the check; after deletion it still fails the history check; output never includes the value |
| Model inventory | Both trusted bundled policies loaded on CPU; 15,705 total parameters each, 8,204 in the deterministic actor; exact bytes and SHA-256 digests recorded in `benchmarks/model-inventory.json` |
| Multimodal planner prototype | Three tensor-contract tests: image/text/state gradient paths, finite bounded horizon outputs, instruction padding/UTF-8 tokens and invalid input/config rejection. No trained weights, live inference, perception or planning success. |
| Actual browser recording | GPU checkpoint moved 2.012 m along world X over 5.760 simulated seconds without reset; headless software-WebGL capture median frame interval 143.9 ms. Raw telemetry is in `benchmarks/presentation-2026-10-15/` |

GitHub Actions passed on the golden-puppy, camera-fix and multimodal-scaffold revision
`af88b87f5fac2cb837128b96142e3df531401751`:
[prototype checks](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38862659843).
The earlier pause-fix revision `a15a687` and presentation revisions `7e86dd3` and `f4280eb` also passed. These hosted runs are
separate evidence from local testing. Later source revisions trigger their own
checks; inspect their recorded conclusions before claiming them passed.

One intermediate run, [`1a1f7c2`](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38823098708),
failed its browser-test step despite the local pass. GitHub exposed only the
step's exit code; downloading detailed logs redirected to a storage host blocked
by this environment. A subsequent local run reproduced an intermittent failure
in the new strict pause assertion: a weighted sum of identical timestamps changed
the final floating-point bit. The render clock now uses a delta interpolation
that stays exactly constant when paused, with a fractional-interval regression
test. The original remote assertion is still unavailable, so that run cannot be
conclusively attributed to the same cause. CI now uses Playwright's GitHub reporter
to expose future failing assertions as check annotations. No assertions or tests
were disabled or loosened.

The [first golden-puppy hosted run](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38861840424)
exposed a second rounding issue: OrbitControls' unbounded target-radius clamp
changed the last floating-point bit of an idle target after follow was disabled.
The check annotation identified the exact assertion. An adapter now preserves
the prior target when there is no interaction/follow and controls report no
change; real pan/orbit/follow updates remain enabled. The strict browser assertion
is retained and an idle/active update regression was added.

## Scope of the evidence

The historical ball-contact and behind-ball successes are short deterministic
trials. The GPU walker failed its ball-contact trial; curious-v2 remains the
default motor in the new RGB hierarchy. Historical follow/push failures remain
visible alongside the later varied-start successes. No learned self-righting or
Jev model is present.

Not run: physical hardware, sim-to-real, rough terrain, adversarial/mutually
untrusted participants, Windows/Apple Silicon installation, a two-hour multi-user
soak test, or fresh-task snapshot restoration after environment publication.
Configuration saving and GitHub publication do not prove snapshot publication.
