# Validation record

Validated on the development Linux machine with Python 3.12.14, Node 24.19.0,
MuJoCo 3.3.7, PyTorch 2.9.0 CPU and optional JAX 0.7.2. Runpod hardware and source
hashes are recorded separately in `benchmarks/runpod-attempt-*/`.

| Check | Observed result |
| --- | --- |
| `ROBOPUP_MJX=1 bash scripts/setup.sh` | Passed frozen Python/npm installation, two checkpoint digest checks, browser build, operator CLI verification and Python tests; repeat runs preserve local outputs |
| `pytest -q` | 16 passed, no skipped/disabled/expected failures; one upstream AnyIO deprecation warning |
| `ruff check src tests scripts` | Passed |
| `npm run build` | Passed; one approximately 523 kB JS bundle produces Vite's size advisory |
| `npm test` | 11 passed: eight browser checks including real geometry/controls, experiment results and policy loading, command grammar, physical CPU-policy ball contact, GPU-policy locomotion, close-up/follow camera and frozen expressions on pause, mobile layout, five simultaneous browsers; three deterministic render timing/reset checks |
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
| Actual browser recording | GPU checkpoint moved 2.012 m along world X over 5.760 simulated seconds without reset; headless software-WebGL capture median frame interval 143.9 ms. Raw telemetry is in `benchmarks/presentation-2026-10-15/` |

GitHub Actions also passed on the earlier code revision
`b7a5976e1049c3077edb734e475a83fe1b270b87`:
[prototype checks](https://github.com/sanjuhs/robo-pup-decision-model/actions/runs/38821508980).
This hosted run is separate evidence from local testing. Later source revisions
trigger their own checks; inspect their recorded conclusions before claiming them passed.

## Scope of the evidence

The ball-contact and behind-ball successes are short deterministic trials, not
general manipulation benchmarks. The GPU-trained walker failed its ball-contact
trial; curious-v2 is the validated ball-demo policy. Follow and push-to-goal
trials failed and remain visible. No learned self-righting or Jev model is present.

Not run: physical hardware, sim-to-real, rough terrain, adversarial/mutually
untrusted participants, Windows/Apple Silicon installation, a two-hour multi-user
soak test, or fresh-task snapshot restoration after environment publication.
Configuration saving and GitHub publication do not prove snapshot publication.
