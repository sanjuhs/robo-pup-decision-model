#!/usr/bin/env bash
# No Runpod account credential is ever passed here. External monitor deletes pod.
set -uo pipefail
mkdir -p /workspace/job-results/runpod
if [[ ${1:-} == --inner ]]; then
  set -e
  printf '%s\n' 'ROBOPUP_STAGE:vlm-bootstrap'
  cd /workspace
  timeout 60s git clone https://github.com/sanjuhs/robo-pup-decision-model.git robopup
  cd robopup
  git checkout --detach "$ROBOPUP_REF"
  python3 - <<'PY'
import os,pathlib,shutil
relative=pathlib.PurePosixPath(os.environ['ROBOPUP_INPUT_DIR'])
if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid public input path')
source=(pathlib.Path.cwd()/relative).resolve()
if not source.is_relative_to(pathlib.Path.cwd()) or not source.is_dir():raise ValueError('Input path must be inside checked-out source')
shutil.copytree(source,'/workspace/vlm-input')
PY
  export PYTHONPATH=/workspace/robopup/src
  export PYTHONUNBUFFERED=1 XDG_CACHE_HOME=/workspace/cache HF_HOME=/workspace/hf-cache
  export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1
  export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30
  apt-get update -qq
  apt-get install -y -qq libegl1 libgl1-mesa-dri
  python3 -c 'import sys,torch,torchvision; assert sys.version_info[:2]==(3,12); assert torch.__version__=="2.9.0+cu129"; assert torchvision.__version__=="0.24.0+cu129"; assert torch.cuda.is_available()'
  python3 -m venv --system-site-packages /workspace/vlm-env
  /workspace/vlm-env/bin/python -m pip install --no-index --require-hashes --no-deps -r configs/gpu-vlm-runtime-direct-cp312.txt
  export PATH=/workspace/vlm-env/bin:$PATH
  sha256sum configs/gpu-vlm-runtime-direct-cp312.txt > /workspace/job-results/runpod/runtime-requirements.sha256
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > /workspace/job-results/runpod/hardware.csv
  python -m pip list --format=json > /workspace/job-results/runpod/packages.json
  python - <<'PY'
import json,pathlib,torch,torchvision,transformers,accelerate
record={'torch':torch.__version__,'torchvision':torchvision.__version__,
 'transformers':transformers.__version__,'accelerate':accelerate.__version__,
 'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
 'compute_capability':list(torch.cuda.get_device_capability()),
 'source_ref':__import__('os').environ['ROBOPUP_REF']}
pathlib.Path('/workspace/job-results/runpod/runtime.json').write_text(json.dumps(record,indent=2)+'\n')
PY
  nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,power.draw --format=csv -l 2 > /workspace/job-results/runpod/gpu-utilization.csv &
  sampler=$!
  trap 'kill "$sampler" 2>/dev/null || true' EXIT
  printf '%s\n' 'ROBOPUP_STAGE:vlm-download-public-checkpoint'
  python scripts/download_vlm.py --output /workspace/vlm-checkpoint
  cp /workspace/vlm-checkpoint/robopup-model-manifest.json /workspace/job-results/runpod/model-manifest.json
  find /workspace/vlm-checkpoint -maxdepth 1 -iname 'LICENSE*' -exec cp {} /workspace/job-results/runpod/ \;
  export ROBOPUP_VLM_MODEL_DIR=/workspace/vlm-checkpoint
  cp -a /workspace/vlm-input /workspace/job-results/runpod/inputs
  printf '%s\n' 'ROBOPUP_STAGE:vlm-real-camera-inference'
  python -u - <<'PY'
import hashlib,json,pathlib,re,time
from PIL import Image
from robopup.vlm_protocol import parse_mission,MissionValidationError
from robopup.vlm_runtime import get_vlm_runtime,VLMRuntimeError
inputs=pathlib.Path('/workspace/vlm-input');out=pathlib.Path('/workspace/job-results/runpod')
data=json.loads((inputs/'prompts.json').read_text());cases=data if isinstance(data,list) else data['cases']
runtime=get_vlm_runtime();results=[];started=time.perf_counter()
for index,case in enumerate(cases):
 identifier=case.get('id',f'case-{index:03}')
 if not isinstance(identifier,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',identifier):raise ValueError('Invalid case ID')
 filename=case['image']
 if not isinstance(filename,str) or pathlib.Path(filename).name!=filename or not filename.endswith('.png'):raise ValueError('Invalid case image path')
 image_path=inputs/filename
 record={'id':identifier,'instruction':case['instruction'],'energy':case.get('energy',.8),
  'held':case.get('held',False),'image':filename,'image_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
  'generative_inference_requested':True,'actual_generated_response_received':False,
  'physical_task_tested':False,'valid_mission':False}
 try:
  with Image.open(image_path) as image:generation=runtime.generate(image,record['instruction'],record['energy'],record['held'])
  record['generation']=generation.as_dict();record['actual_generated_response_received']=True
  record['mission']=parse_mission(generation.raw_text).as_dict();record['valid_mission']=True
 except (VLMRuntimeError,MissionValidationError) as error:record['error']=str(error)
 (out/f'case-{index:03}-{identifier}.json').write_text(json.dumps(record,indent=2)+'\n')
 results.append(record)
 print('ROBOPUP_STAGE:'+json.dumps({'case':identifier,'valid_mission':record['valid_mission']}),flush=True)
summary={'schema':'robopup-real-vlm-batch-v1','cases':len(cases),'valid_missions':sum(r['valid_mission'] for r in results),
 'elapsed_seconds':time.perf_counter()-started,'physical_task_tested':False,'final_runtime_status':runtime.status(),
 'results':results,'note':'Actual camera image inference; generated plans are not evidence of completed physical tasks.'}
(out/'probe-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
  printf '%s\n' 'ROBOPUP_STAGE:vlm-closed-loop-native-physics'
  set +e
  timeout 11m python -u scripts/evaluate_vlm_mission.py --out /workspace/job-results/runpod/mission --seed 7 --sim-seconds 360 --wall-seconds 600
  mission_status=$?
  set -e
  export ROBOPUP_MISSION_STATUS="$mission_status"
  python - <<'PY'
import json,os,pathlib
root=pathlib.Path('/workspace/job-results/runpod')
static=json.loads((root/'probe-summary.json').read_text())
record={'static_cases':static['cases'],'valid_static_missions':static['valid_missions'],
 'closed_loop_exit_code':int(os.environ['ROBOPUP_MISSION_STATUS']),
 'all_static_valid':static['cases']==static['valid_missions']}
(root/'combined-status.json').write_text(json.dumps(record,indent=2)+'\n')
if not record['all_static_valid'] or record['closed_loop_exit_code']!=0:raise SystemExit(2)
PY
  printf '%s\n' 'ROBOPUP_STAGE:vlm-inference-complete'
  exit 0
fi
timeout 23m bash "$0" --inner 2>&1 | tee /workspace/job-results/runpod/job.log
job_status=${PIPESTATUS[0]}
export ROBOPUP_JOB_STATUS="$job_status"
python3 - <<'PY'
import base64,hashlib,io,json,os,pathlib,tarfile,time
root=pathlib.Path('/workspace/job-results/runpod')
(root/'job-status.json').write_text(json.dumps({'exit_code':int(os.environ['ROBOPUP_JOB_STATUS'])})+'\n')
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w:gz') as archive:archive.add(root,arcname='runpod')
data=stream.getvalue();encoded=base64.b64encode(data).decode();chunks=[encoded[i:i+6000] for i in range(0,len(encoded),6000)]
print('ROBOPUP_BUNDLE_SHA256:'+hashlib.sha256(data).hexdigest(),flush=True)
for index,chunk in enumerate(chunks):
 print(f'ROBOPUP_BUNDLE:{index:05}/{len(chunks):05}:{chunk}',flush=True);time.sleep(.03)
PY
printf 'ROBOPUP_DONE:%s\n' "$job_status"
sleep infinity
