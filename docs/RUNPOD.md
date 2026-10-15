# Runpod: one authorized GPU experiment

The total GPU experimentation budget is **$30**. The user subsequently authorized paid work within this budget and requested
one training experiment. One initial MJX/SB3 interface attempt failed; its pod
was deleted and the same experiment was retried after a regression-tested fix.
No additional policy experiments or persistent services are authorized by this runbook.
The checked-in scripts do not provision pods; `gpu_job.sh` runs only inside an
already authorized, externally monitored disposable pod.

## Credentials and read-only access

`runpodctl` v3.0.1 is installed at `/workspace/.tools/runpodctl`, using the official
release's SHA-256 checksum. Reinstall or verify idempotently:

```bash
python3 scripts/install_runpodctl.py
/workspace/.tools/runpodctl --version
python3 scripts/runpod_preflight.py
```

The preflight only queries account identity and the GPU catalog. It prints neither
credentials nor the account ID. It reports failures without dumping response bodies.
It reads `RUNPOD_API_KEY` from the process environment and never saves its value.
Use the cloud environment's secure secret entry, scoped to `api.runpod.io` and
`rest.runpod.io`; never put the key in source, `.env` committed to Git, CLI flags,
shell history, screenshots, or Markdown. The chat-supplied key was not copied into
any repository file, command, or saved script. Rotate a chat-shared credential
before using it as a long-lived deployment secret.

The environment draft contains the secret requirement and added API domains.
Draft saving does not inject the value, update runtime networking, or publish the
environment. Review/save the settings and publish as indicated by the product.
For future tasks, enter the key securely there and rerun preflight. The current
session separately verified account access and CLI pod listing using hidden,
ephemeral input. `api.github.com` was also requested
for repository/CLI metadata; Git HTTPS and release downloads already work.

CLI commands such as `pod list`, `gpu list`, and `user` are read-only. Do not run
`doctor` interactively with a production key or put `--apiKey` in a command line.
Only print sanitized output when collecting account evidence.

## Execution plan and lifecycle limits

These are the bounds used for the authorized experiment; do not automatically rerun it:

- One on-demand RTX 5090, one GPU; no spot retry, fallback GPU, paid endpoint,
  persistent network volume, or automatic scaling.
- At most 25 minutes per disposable attempt, with a $5 experiment allocation
  inside the overall $30 ceiling. Both attempts used a reported $1.19/hour rate.
  Actual elapsed intervals, deletion checks, estimates and billing evidence belong
  in `benchmarks/`; do not confuse catalog rates with final receipts.
- Provider image: `runpod/pytorch:1.2.0-cu1290-torch290-ubuntu2404`,
  requested CUDA capability >=12.9, one RTX 5090. The first host reported driver
  595.91.07 and 32,607 MiB GPU memory. JAX CUDA wheels are pinned in `uv.lock`.
- Native CPU results are the baseline. The custom MJX adapter uses the same robot
  and policy interface; Playground/Warp is not integrated. GPU physics is in JAX;
  the small PPO network is optimized on CPU.
- Record GPU model, driver, CUDA/PyTorch/JAX versions, image digest, setup time,
  compile time, throughput, evaluation, billed duration, storage, and actual receipt.
- Before launch, establish a provider-side lifecycle limit or an independent
  termination monitor and a verified manual cleanup path. A training timeout alone
  does not prevent an abandoned pod from billing. If lifecycle protection cannot
  be verified, do not launch.
- Export checkpoints, terminate the approved pod, and verify billable storage is
  removed or separately approved. Do not delete pre-existing user resources.

Read-only Runpod access and the CLI were verified. The selected pod response
confirmed $1.19/hour. See `BENCHMARKS.md` for actual training and physics results.
The external monitor retrieves checkpoint bundles, verifies their SHA-256, deletes
only its own pod, and verifies absence. No persistent volume or public port was
created. The monitor time limit is an external safeguard, not a provider-enforced
spending cap. `gpu_job.sh` also limits its inner job to 19 minutes and training to
10 minutes; finishing a process by itself does not stop billing.

## Reproduce the current CPU prototype on an approved pod

After approval and provisioning through the Runpod console, open its terminal:

```bash
cd /workspace
git clone https://github.com/sanjuhs/robo-pup-decision-model.git
cd robo-pup-decision-model
bash scripts/setup.sh
bash scripts/start.sh
```

Python 3.11–3.13 and a supported Node installation are required in the chosen image.
The setup script uses frozen dependencies, preserves TLS/checksum verification,
and installs CPU PyTorch. The native application was tested in the onboarding machine. The headless
MJX training path was separately executed in the Runpod image above; the browser
was not exposed from the rented pod. Use an SSH tunnel to the loopback-only server;
do not expose an unauthenticated public port.

## Reproduce the MJX training path

Only on an authorized GPU host, with a fresh output directory:

```bash
uv sync --frozen --extra mjx --extra cuda
.venv/bin/python -m robopup.train_mjx --require-gpu --envs 128 --steps 1048576 --max-minutes 10 --out artifacts/runpod/new-run
```

This installs CPU PyTorch for optimization and JAX CUDA libraries for **GPU
physics**. Keep the native laptop environment separate. The runner records
compilation time, finite-state and forced-timeout/reset probes, physics throughput,
source hashes, checkpoints, logs, and a five-seed evaluation on native MuJoCo.
A saved policy can then run in the CPU browser arena. `--probe-only` measures
physics without starting learning. A GPU run requires `--require-gpu` so an
accidental CPU fallback fails early.

The checked-in `gpu_job.sh` contains the tested inside-container sequence and
checkpoint export protocol. The account API key is never passed into that job.
It assumes an external lifecycle monitor; never launch it as an unattended
billing guard by itself. The original hosted experiment is complete only when
its lifecycle record confirms deletion and its result JSON confirms success.
