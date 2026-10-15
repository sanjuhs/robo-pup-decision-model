#!/usr/bin/env bash
# Run inside an already authorized, externally monitored disposable GPU pod.
# This script never receives the account API key and does not provision anything.
set -uo pipefail
printf '%s\n' 'ROBOPUP_STAGE:bootstrap'
timeout 19m bash -euo pipefail -c '
  cd /workspace
  python3 -m pip install uv==0.12.23
  git clone https://github.com/sanjuhs/robo-pup-decision-model.git robopup
  cd robopup
  git checkout --detach "$ROBOPUP_REF"
  export UV_CACHE_DIR=/workspace/uv-cache
  export MPLCONFIGDIR=/workspace/matplotlib-cache
  export XDG_CACHE_HOME=/workspace/cache
  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PYTHONUNBUFFERED=1
  uv sync --frozen --extra mjx --extra cuda
  mkdir -p artifacts/runpod
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > artifacts/runpod/hardware.csv
  printf "%s\n" "ROBOPUP_STAGE:training-entrypoint"
  .venv/bin/python -m robopup.train_mjx --require-gpu --envs 128 --steps 1048576 --max-minutes 10 --out artifacts/runpod/5090-once
'
job_status=$?
export ROBOPUP_JOB_STATUS="$job_status"
if [[ -d /workspace/robopup/artifacts/runpod ]]; then
  python3 - <<'PY'
import base64, hashlib, io, json, os, pathlib, tarfile, time
root=pathlib.Path('/workspace/robopup/artifacts/runpod')
(root/'job-status.json').write_text(json.dumps({'exit_code':int(os.environ['ROBOPUP_JOB_STATUS'])}))
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:
    archive.add(root,arcname='runpod')
data=stream.getvalue()
encoded=base64.b64encode(data).decode()
chunks=[encoded[i:i+6000] for i in range(0,len(encoded),6000)]
print('ROBOPUP_BUNDLE_SHA256:'+hashlib.sha256(data).hexdigest(),flush=True)
for index,chunk in enumerate(chunks):
    print(f'ROBOPUP_BUNDLE:{index:05}/{len(chunks):05}:{chunk}',flush=True)
    time.sleep(.03)
PY
fi
printf 'ROBOPUP_DONE:%s\n' "$job_status"
# The external lifecycle monitor retrieves results and deletes this pod. Keep
# logs available until it does; never treat process completion as billing cleanup.
sleep infinity
