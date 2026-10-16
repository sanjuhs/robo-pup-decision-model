# Validation record

Validated on the development Linux machine with Python 3.12.14, Node 24.19.0,
MuJoCo 3.3.7, PyTorch 2.9.0 CPU and optional JAX 0.7.2. Runpod hardware and source
hashes are recorded separately in `benchmarks/runpod-attempt-*/`.

## Current GPU-trained v4 release

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
Hosted CI for this new default is recorded after completion below.

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
