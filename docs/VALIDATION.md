# Validation record

Validated on the development Linux machine with Python 3.12.14, Node 24.19.0,
MuJoCo 3.3.7, PyTorch 2.9.0 CPU and optional JAX 0.7.2. Runpod hardware and source
hashes are recorded separately in `benchmarks/runpod-attempt-*/`.

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

The ball-contact and behind-ball successes are short deterministic trials, not
general manipulation benchmarks. The GPU-trained walker failed its ball-contact
trial; curious-v2 is the validated ball-demo policy. Follow and push-to-goal
trials failed and remain visible. No learned self-righting or Jev model is present.

Not run: physical hardware, sim-to-real, rough terrain, adversarial/mutually
untrusted participants, Windows/Apple Silicon installation, a two-hour multi-user
soak test, or fresh-task snapshot restoration after environment publication.
Configuration saving and GitHub publication do not prove snapshot publication.
