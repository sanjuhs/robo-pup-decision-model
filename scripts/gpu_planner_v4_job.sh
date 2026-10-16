#!/usr/bin/env bash
# Execute only inside an authorized pod with the external deletion monitor active.
# No account credential is passed to this job.
set -uo pipefail
mkdir -p /workspace/job-results/runpod
printf '%s\n' 'ROBOPUP_STAGE:planner-bootstrap'
timeout 35m bash -euo pipefail -c '
  cd /workspace
  python3 -c "import urllib.request; r=urllib.request.urlopen(\"https://files.pythonhosted.org/packages/b3/7d/01e62f59e4166af1be6238d9f2b7f3453630c51b8044b412d05b5606532a/absl_py-2.5.1-py3-none-any.whl\",timeout=20); print(r.status)"
  timeout 60s git clone https://github.com/sanjuhs/robo-pup-decision-model.git robopup
  cd robopup
  git checkout --detach "$ROBOPUP_REF"
  mkdir -p /workspace/job-results/runpod/planner-v4
  python3 -c "import sys; assert sys.version_info[:2]==(3,12), sys.version"
  export MPLCONFIGDIR=/workspace/matplotlib-cache XDG_CACHE_HOME=/workspace/cache PYTHONUNBUFFERED=1
  apt-get update -qq
  apt-get install -y -qq libegl1 libgl1-mesa-dri
  python3 -m venv --system-site-packages /workspace/planner-env
  /workspace/planner-env/bin/python -m pip install --no-index --require-hashes --no-deps -r configs/gpu-runtime-direct-cp312.txt
  export PYTHONPATH=/workspace/robopup/src
  sha256sum configs/gpu-runtime-direct-cp312.txt > /workspace/job-results/runpod/runtime-requirements.sha256
  export PATH=/workspace/planner-env/bin:$PATH
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /workspace/job-results/runpod/hardware.csv
  python -m pip list --format=json > /workspace/job-results/runpod/packages.json
  python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name())"
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 5 > /workspace/job-results/runpod/gpu-utilization.csv &
  sampler=$!
  trap "kill $sampler 2>/dev/null || true" EXIT
  printf "%s\n" "ROBOPUP_STAGE:planner-data"
  python -m robopup.planner_data --out artifacts/planner-gpu-data-v4 --scenes 6000 --seed 6119 --rollout-steps 20 --static-fraction 0.5
  cp artifacts/planner-gpu-data-v4/metadata.json /workspace/job-results/runpod/training-data.json
  printf "%s\n" "ROBOPUP_STAGE:planner-training"
  timeout 25m python -u -m robopup.train_gpu_planner --data artifacts/planner-gpu-data-v4 --out /workspace/job-results/runpod/planner-v4/weights --steps 8000 --batch-size 128 --device cuda --architecture v4 --initialize demo/arena-planner-v3 --seed 521
  printf "%s\n" "ROBOPUP_STAGE:planner-complete"
' 2>&1 | tee /workspace/job-results/runpod/job.log
job_status=${PIPESTATUS[0]}
export ROBOPUP_JOB_STATUS="$job_status"
if [[ -d /workspace/job-results/runpod ]]; then
python3 - <<'PY'
import base64,hashlib,io,json,os,pathlib,tarfile,time
root=pathlib.Path('/workspace/job-results/runpod')
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
