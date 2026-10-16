#!/usr/bin/env bash
# Execute only inside an authorized pod with the external deletion monitor active.
# No account credential is passed to this job.
set -uo pipefail
printf '%s\n' 'ROBOPUP_STAGE:planner-bootstrap'
timeout 35m bash -euo pipefail -c '
  cd /workspace
  python3 -m pip install uv==0.12.23
  git clone https://github.com/sanjuhs/robo-pup-decision-model.git robopup
  cd robopup
  git checkout --detach "$ROBOPUP_REF"
  mkdir -p artifacts/runpod/planner-v3
  export MPLCONFIGDIR=/workspace/matplotlib-cache XDG_CACHE_HOME=/workspace/cache PYTHONUNBUFFERED=1
  apt-get update -qq
  apt-get install -y -qq libegl1 libgl1-mesa-dri
  python3 -m venv --system-site-packages /workspace/planner-env
  python3 scripts/export_gpu_runtime.py --out /workspace/gpu-runtime.txt
  /workspace/planner-env/bin/python -m pip install --require-hashes --no-deps -r /workspace/gpu-runtime.txt
  /workspace/planner-env/bin/python -m pip install --no-deps -e .
  export PATH=/workspace/planner-env/bin:$PATH
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > artifacts/runpod/hardware.csv
  python -m pip list --format=json > artifacts/runpod/packages.json
  python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > artifacts/runpod/gpu-utilization.csv &
  sampler=$!
  trap "kill $sampler 2>/dev/null || true" EXIT
  printf "%s\n" "ROBOPUP_STAGE:planner-data"
  python -m robopup.planner_data --out artifacts/planner-gpu-data --scenes 6000 --seed 5041 --rollout-steps 20
  cp artifacts/planner-gpu-data/metadata.json artifacts/runpod/training-data.json
  printf "%s\n" "ROBOPUP_STAGE:planner-training"
  timeout 25m python -u -m robopup.train_gpu_planner --data artifacts/planner-gpu-data --out artifacts/runpod/planner-v3/weights --steps 8000 --batch-size 128 --device cuda
  printf "%s\n" "ROBOPUP_STAGE:planner-complete"
'
job_status=$?
export ROBOPUP_JOB_STATUS="$job_status"
if [[ -d /workspace/robopup/artifacts/runpod ]]; then
python3 - <<'PY'
import base64,hashlib,io,json,os,pathlib,tarfile,time
root=pathlib.Path('/workspace/robopup/artifacts/runpod')
(root/'job-status.json').write_text(json.dumps({'exit_code':int(os.environ['ROBOPUP_JOB_STATUS'])}))
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:archive.add(root,arcname='runpod')
data=stream.getvalue();encoded=base64.b64encode(data).decode();chunks=[encoded[i:i+6000] for i in range(0,len(encoded),6000)]
print('ROBOPUP_BUNDLE_SHA256:'+hashlib.sha256(data).hexdigest(),flush=True)
for index,chunk in enumerate(chunks):
 print(f'ROBOPUP_BUNDLE:{index:05}/{len(chunks):05}:{chunk}',flush=True);time.sleep(.03)
PY
fi
printf 'ROBOPUP_DONE:%s\n' "$job_status"
sleep infinity
