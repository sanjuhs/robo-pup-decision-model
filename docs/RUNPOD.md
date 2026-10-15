# Runpod: prepared, no paid resources provisioned

The total GPU experimentation budget is **$30**. The user requires explicit
approval before paid provisioning. No repository script creates a pod, volume,
serverless endpoint, or paid service. A key or a successful authentication check
does not authorize spending.

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
Then rerun preflight in the updated runtime. `api.github.com` was also requested
for repository/CLI metadata; Git HTTPS and release downloads already work.

CLI commands such as `pod list`, `gpu list`, and `user` are read-only. Do not run
`doctor` interactively with a production key or put `--apiKey` in a command line.
Only print sanitized output when collecting account evidence.

## Approval proposal: first GPU experiment

This is a plan for review, not an executable provisioning instruction:

- One on-demand RTX 5090, one GPU; no spot retry, fallback GPU, paid endpoint,
  persistent network volume, or automatic scaling.
- First session: at most 30 minutes from pod creation, **maximum $5 all-in**,
  counted within the overall $30 budget. Obtain the live GPU, storage, and transfer
  quote before asking for final approval; use the smaller time allowed by that quote.
- Use a provider-maintained PyTorch/CUDA image supporting Blackwell (`sm_120`).
  CUDA 12.8/PyTorch 2.9 is the proposed compatibility path, not a validated image pin.
- Measure native CPU physics/PPO first. Then separately plan Playground MJX/Warp;
  that path is not implemented in this custom environment yet.
- Record GPU model, driver, CUDA/PyTorch/JAX versions, image digest, setup time,
  compile time, throughput, evaluation, billed duration, storage, and actual receipt.
- Before launch, establish a provider-side lifecycle limit or an independent
  termination monitor and a verified manual cleanup path. A training timeout alone
  does not prevent an abandoned pod from billing. If lifecycle protection cannot
  be verified, do not launch.
- Export checkpoints, terminate the approved pod, and verify billable storage is
  removed or separately approved. Do not delete pre-existing user resources.

No live rate, availability, account balance, or Runpod access is claimed until the
secure binding and network policy are active and read-only checks succeed.

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
and installs CPU PyTorch. This exact application path was tested in the onboarding
machine, **not** in a Runpod pod. Use an SSH tunnel to the loopback-only server;
do not expose an unauthenticated public port.

## CUDA optimization experiment (unverified)

Keep the tested CPU environment intact. In a separate virtual environment, install
PyTorch 2.9.0 from its official `cu128` wheel index, then the pinned application
dependencies. Check `torch.cuda.is_available()`, the reported device name and
capability, and a real CUDA matrix operation before requesting `--device cuda`.
The checked-in `uv.lock` selects CPU wheels; ordinary `uv sync` will restore that
CPU selection. No CUDA setup is currently claimed tested or fully locked.

Even a successful CUDA PPO run still uses native **CPU physics**. A true GPU
physics path needs MJX/Warp/Playground integration, compatible robot assets, and
new evaluation. Do not describe CPU MuJoCo plus CUDA optimization as GPU physics.
