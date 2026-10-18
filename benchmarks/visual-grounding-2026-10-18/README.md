# Actual CPU counterfactual evaluation — 2026-10-18

**Qwen failed all four paired visual-decision checks.** This is new actual
inference, not a response fixture or a simulated model. The existing visibility
gate independently accepted six visible-target fetch candidates and abstained
on two unsupported candidates. No weights, production prompts or thresholds
were changed. These measurements do not establish general visual reliability.

## Executed

Eight genuine calls used the pinned Qwen3-VL-2B-Instruct revision
`89644892e4d85e24eaac8bacfd4f463576704203`, local Transformers, CPU bfloat16,
two model threads, deterministic decoding and a 300-second cooperative timeout.
All eight outputs parsed as supported mission JSON; none timed out or truncated.
There was one generation per case, no retries, no action execution and no GPU
allocation. Seven unique RGB images underlie the eight cases: one visible view
is reused with a different prompt. This is not eight independent visual samples.

| Intervention | Expected decisions | Actual raw-model decisions | Both correct |
| --- | --- | --- | --- |
| Target missing | fetch / look | fetch / fetch | No |
| Ball changes sides | left / right | both directions / both directions | No |
| Ball moves behind occluder | fetch / look | fetch / fetch | No |
| Rotated and pitched viewpoint, ball changes sides | left / right | look / look | No |

There were **0/4 semantic decision changes**, **2/4 executable-plan changes**,
**2/8 correct branch decisions**, and **1/8 strict instruction-compliant cases**.
The extra approach before fetch on the visible occlusion case is counted as a
correct fetch decision but fails the stricter requested-sequence check. This
explicit distinction avoids confusing instruction following with visual branch
selection. Both position plans contained left and right turns; they are conflicts.
The unfamiliar-view outputs contained only look, without the requested final stop.

The guard's fixed-fetch confusion counts were **6 correct accepts, 2 correct
abstentions, 0 unsafe accepts and 0 false abstentions**. Deduplicating the repeated
RGB view gives 5 correct accepts and 2 correct abstentions on 7 unique images.
The gate has no semantic absence/rejection class: rejecting an ungrounded fetch
is an abstention. It cannot establish correct left/right decisions. Its classifier
and thresholds remain the original limited arena-colour system.

Median wall time per model call was **39.775 s**, range **36.588–63.134 s**;
summed call wall time was **343.611 s**, including **4.299 s** initial model load.
Median guard observation time in the final reproduction was **32.063 ms**, with
one Torch thread and no concurrent VLM inference. These are planning/gate
measurements, not physical mission latency, interactive GPU performance or
control-loop speed. See [timing-summary.json](timing-summary.json).

## Evidence and reproduction

- [pairs.json](pairs.json) contains image hashes, offline simulator pixel truth,
  expected decisions and independent gate reports. RGB and learned-mask PNGs are
  beside it; labels never enter model or guard inference.
- [model-results.json](model-results.json) applies the final scorer to the actual
  recorded generations. `new_inference=false` describes **re-scoring**, not a lack
  of the eight actual calls. Original `*-generation.json` records retain their
  initial execution-time scorer fields; use the final aggregate for final scores.
- [runtime-before.json](runtime-before.json), [runtime-after.json](runtime-after.json),
  [model-manifest.json](model-manifest.json) and `inference-configuration.json`
  preserve actual model provenance and the initially executed script/scorer hashes.
  `configuration.json` records the final rendering/re-scoring source hashes. The
  scorer was refined during the run; raw model text and all images were unchanged.
- [archived-model-results.json](archived-model-results.json) reuses the older
  three genuine GPU image-ablation outputs: 0/2 both-correct pairs, 1/3 unique
  correct cases. Those pairs share a baseline and are separate from new CPU calls.
- [legacy-guard-reproduction.json](legacy-guard-reproduction.json) reproduces the
  original 17 fresh views: 18 positives and 50 negatives, no classification errors.
  Its old renderer images are not duplicated here.
- [checks-initial-proxy-failure.txt](checks-initial-proxy-failure.txt) retains the
  first full-suite failure: ambient SOCKS proxy configuration required an optional
  dependency in an unrelated HTTP test. Rerunning with proxy variables removed
  passed; the test and application were not changed to hide the failure.
- [checks-final.txt](checks-final.txt) records the subsequent full-suite result.
  Ruff and the public-secret/history scanner also passed before publishing.

The frozen image hashes reproduced exactly across fresh renders. To recompute
these scores without loading Qwen, use a new directory:

```bash
uv run --no-sync python scripts/evaluate_visual_grounding.py \
  --dataset benchmarks/visual-grounding-2026-10-18 \
  --replay-model-results benchmarks/visual-grounding-2026-10-18 \
  --replay-original --out artifacts/paired-score-reproduction
```

For new actual inference, substitute `--run-model` for `--replay-model-results`
and its argument. See [the protocol](../../docs/VISUAL_GROUNDING_EVALUATION.md).
CI only renders and scores the guard/old archive; it does not download or run Qwen.

## Remaining scope

Unexecuted: repeated generations, new GPU latency measurements, model/threshold
training, independently held-out renderers, new object appearances or distractors,
real-camera images, hardware and physical mission verification for these scenes.
The viewpoint is outside the old eye-probe pose grid, not proven unseen in model
training. Keep these inputs as regression diagnostics; use separate data for
future tuning and independent evaluation. More same-renderer cases alone cannot
establish robust visual reasoning.
