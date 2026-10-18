# Paired visual-grounding evaluation

This evaluates **raw Qwen decisions** and the **existing eye gate** independently.
It does not train either model, alter gate thresholds, execute actions, or claim
physical success. The runner uses `VLMRuntime.generate`, the production prompt,
strict `parse_mission`, and `EyeGrounding` / `require_visible_target` unchanged.
The existing eye evaluator provides rendering, settling and offline labels;
its only extension is an offline horizontal pixel centroid for position truth.

## Reproduce without model compute

```bash
uv sync --extra dev --extra mjx --extra vlm
MUJOCO_GL=egl uv run --no-sync python scripts/evaluate_visual_grounding.py \
  --out artifacts/paired-guard --replay-original
```

A new output directory is mandatory. This renders eight 384×384 RGB images in
four pairs and scores fixed fetch candidates through the gate. It optionally
re-scores the **existing** three genuine Qwen image-ablation outputs, checking
image hashes and identical instructions; replay is explicitly not new inference.
CI runs this cheap path, not Qwen. No duplicated copy of the old dataset is needed.

## Run the actual model on the frozen inputs

Download the pinned checkpoint explicitly with the existing downloader if needed:

```bash
uv run --no-sync python scripts/download_vlm.py
ROBOPUP_VLM_DTYPE=bfloat16 ROBOPUP_VLM_TIMEOUT_S=300 \
  uv run --no-sync python scripts/evaluate_visual_grounding.py \
  --dataset artifacts/paired-guard --out artifacts/paired-qwen --run-model \
  --replay-original
```

`--dataset` reuses the frozen manifest and verifies every image checksum before
copying/inference. Without it, the same fixed recipes render again. There is no
automatic download, GPU allocation, fallback planner or response fixture. Local
weights or an explicitly configured existing runtime provider are necessary.
Bfloat16 is an explicit CPU memory choice, not the default float32 runtime.
Raw generation, parser failures, runtime identity, timing and image hashes are
saved incrementally. Interrupted runs retain completed outputs; they are not
silently counted as successful pairs. Use a new output directory for each run.

Saved raw generations can also be re-scored without compute, checking that the
image hashes, tasks, instructions and fixed state match the frozen dataset:

```bash
uv run --no-sync python scripts/evaluate_visual_grounding.py \
  --dataset artifacts/paired-guard --out artifacts/paired-rescored \
  --replay-model-results artifacts/paired-qwen
```

This labels its output as replay and retains missing/invalid generations as
failures. `--dataset` also reuses its recorded gate observations; it does not
pretend to have re-run the guard. Re-render without that option to measure it again.

## Scene design and truth boundary

Within each pair, text, energy (0.9), held state (false), rendering seed and
non-target scene configuration are fixed. Only the declared intervention changes.
Labels come from renderer geometry IDs **offline**, never from the gate or Qwen.
No labels, coordinates, centroids, expected actions, or gate scores enter inference.

| Pair | Intervention | Expected raw decision |
| --- | --- | --- |
| Missing object | Ball in arena vs ball relocated below the floor, out of scene | fetch vs look |
| Changed position | Ball moves between image-left and image-right | turn_left vs turn_right |
| Occlusion | Reuse the existing before/behind-obstacle ball-position recipe | fetch vs look |
| Unfamiliar viewpoint | Fixed 147° yaw, −12° pitch and translated robot; ball changes sides | turn_left vs turn_right |

The unfamiliar pose is outside the old eye-probe pose grid, not proven unseen in
Qwen pretraining or the compact classifier's training data. The occlusion pair
changes target position to put it behind an existing obstacle; it is not a study
isolating occluder movement. The missing object is moved below the arena rather
than removed from the compiled physics model. Images still use the original
renderer, colours and object identities. This is a small developmental diagnostic,
not a held-out statistical benchmark or real-camera generalization result.
The eight cases contain seven unique RGB images: the visible missing-object
baseline is reused with a different instruction in the position pair. Guard case
counts therefore include one repeated view and are not independent samples.

Visible truth requires at least 64 ball pixels, matching the old evaluator.
Position cases additionally require a centroid at least 10% of image width away
from the centre. Generation fails early if a recipe does not produce different
truth decisions or produces identical images. These checks prevent mislabeled
counterfactuals from quietly entering the score. Tiny/partial targets near that
threshold remain a labeling limitation.

## Scoring and interpretation

Raw model scores include unique-case decision accuracy, strict instruction-following
accuracy, schema-valid pairs, decision-change pairs and **both-sides-correct** pairs. It also reports executable-plan changes,
so different but equally incorrect step sequences are not concealed by the
semantic choice score. A changed answer alone is not success. Summaries
are ignored; executable steps are retained. A look preamble followed by fetch
still counts as fetch, exposing the original failure instead of rewarding its
first step. Contradictory choices, extraneous actions, a missing final stop,
invalid JSON and missing generations cannot earn a strict correct case. The semantic
choice remains separately visible when the model makes extra moves or omits stop;
`correct_decision_cases` measures branch selection and `correct_cases` additionally
requires instruction compliance. Tests use
controlled outputs only to verify these contracts, not as model evidence.

Position instructions test a decision the schema can express; unchanged fetch
commands across two visible ball positions would not test spatial understanding.
These are requested turn decisions, not proof of executed turning or navigation.

The guard is tested on **fixed fetch candidates in every image**, independent of
which action Qwen proposes. This prevents a look/stop output from hiding an unsafe
fetch acceptance. It tests visibility suitability only, not collision safety or
correct spatial decisions. The existing gate cannot check left/right: untargeted
turns are allowed. It also cannot semantically certify absence. Its rejection is
an **ungrounded abstention**, not a claim that the ball does not exist.

| Ground truth | Gate accepts fetch | Gate rejects as ungrounded |
| --- | --- | --- |
| Ball visible | Correct accept | False abstention |
| Ball not visible | Unsafe accept | Correct abstention |

The held-ball exception is tested separately in unit tests; scene pairs all use
held=false. Future target steps require fresh activation images in production;
this evaluator does not misleadingly score later mission steps against a stale
frame. Gate latency and model generation latency are separate measurements.
One deterministic generation per image is not a repeatability estimate, and
model waiting time is not control-loop latency.

Executed evidence and exact results are linked from `benchmarks/visual-grounding-2026-10-18/README.md`.
