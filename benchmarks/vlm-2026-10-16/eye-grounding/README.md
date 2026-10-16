# Eye-image visibility guard evidence

This guard reuses the exact **821 trained pixel-classifier parameters** from the
checksum-verified v4 checkpoint (SHA-256
`a464d92e975ee86ae297ddc23ddab10b52c2f83460550468317b37f3f2b1ae89`).
Only the RGB image enters inference. It does not read MuJoCo objects,
segmentation, coordinates, or the VLM's claimed observations. It does not
project overhead image coordinates into a perspective camera.

The seven `gpu-*-rgb.png` files are exact pixel copies of head-camera inputs
retrieved from the actual RTX 4090 VLM experiment, including its blank and
turned counterfactuals and three mission frames. Original artifact paths and
image hashes are retained per record in [probe.json](probe.json). Their
`*-learned-mask.png` files are this trained classifier's predictions, not
simulator labels. Mask palette: black=unqualified/background, red=ball,
orange=crate, blue=home, purple=beacon.

These recorded images were development probes. The abstention settings were
then fixed at posterior ≥ 0.90 and a four-connected region of at least 64
pixels at 384×384 (scaled by area, with a four-pixel floor). A second probe
rendered 17 fresh views: three new robot positions at five yaws, plus a paired
ball placement before and behind a real obstacle. Offline MuJoCo segmentation
counts label those fresh images **only for evaluation**; live inference never
receives them.

At an offline visibility label of at least 64 true object pixels, the fresh
probe had **18 positive and 50 negative object/view checks**, with zero false
positives or false negatives in that small set. The start image grounded ball,
crate and beacon. Blank, turned and fully occluded images grounded no target,
and an immediate fetch was rejected. This is evidence within the same arena
colours and renderer, not generalization to realistic objects, new textures,
arbitrary lighting, tiny/distant targets, or every camera angle. Classifier
posteriors are not statistically calibrated probabilities.

The guard checks the immediate action when it activates. Later actions need a
fresh frame. Look and turns remain possible when a target is ungrounded;
held-ball proprioception permits returning an already-held ball. Visibility
does not prove grasping, completion, or that a repeated fetch is necessary.

Reproduce without replacing these records:

```bash
uv run python scripts/evaluate_eye_grounding.py \
  --out artifacts/eye-grounding-reproduction
```

The evaluator defaults to EGL before importing MuJoCo, requires an explicit new
output path, and refuses to overwrite existing evidence. It loads the seven
recorded RGB images from this folder and recreates the fresh scene recipes.
An actual reproduction matched all 24 RGB images, learned masks and visibility
reports exactly, and rejected reuse of its output path; see
[reproduction-verification.json](reproduction-verification.json).
An initial ad hoc launch omitted EGL before importing MuJoCo and failed during
headless renderer startup; [initial-headless-startup-failure.json](initial-headless-startup-failure.json)
retains that failure. The successful rerun changed only the rendering backend,
not weights or thresholds.
