# Actual RTX 4090 VLM experiment

The pretrained Qwen3-VL-2B-Instruct model generated actual plans from attached
robot-eye PNGs. It has **2,127,532,032 parameters**, uses BF16 here and is pinned
to revision `89644892e4d85e24eaac8bacfd4f463576704203` under Apache-2.0.
No foundation-model training or fine-tuning was performed.

The genuine closed-loop **fetch → hop → stop** mission passed all 12 recorded
checks on seed 7 using three fresh eye/model calls: **23.4518 wall seconds** and
**19.98 execution simulation seconds**. Simulation was frozen while the model
thought. MuJoCo physics and PPO residuals ran on CPU; the VLM ran on the RTX 4090.
This is one development trial, not a general task-success or browser-latency
benchmark. [Raw mission result](mission/result.json).

The static probes expose important failures:

| Development case | Valid JSON/schema | Declared expectation | Whole-request review |
| --- | --- | --- | --- |
| Fetch, hop, stop | Yes | Exact sequence passed | Matches |
| Old history-only after-fetch prompt | Yes | Exact sequence failed | Repeats completed fetch |
| Conditional, ball-visible image | Yes | First action failed | Returns look, fetch, stop |
| Conditional, blank image | Yes | First action passed | Incorrectly still includes fetch |
| Conditional, turned image | Yes | First action passed | Incorrectly still includes fetch |
| Rest, fetch, stop | Yes | Exact sequence passed | Matches |
| Push crate, hop, stop | Yes | Exact sequence passed | Matches |

**Seven valid JSON plans does not mean seven successful instructions.** The
literal declared checks pass 5/7, including two weak first-action checks. On
descriptive review, only three complete plans match the full request; that
stricter review is not a preregistered metric. All three conditional image cases
produce the **identical raw response**, so reliable image-dependent branching is
not demonstrated. Blank/turned views still provoke an unseen-ball fetch.

The old history-only replan failure is retained. The successful physical trial
uses the corrected runtime's explicit remaining task list, verified completion
memory and sequence guard; these are different prompts and control boundaries.

The fetch verifier passed when the released ball was **0.5032 m** from home.
The ball subsequently rolled to **0.6720 m** during the hop. Persistent retention
inside the 0.6 m home radius was not a criterion and is not demonstrated. The
physical records retain both positions, contact, one grasp/release, **0.3558 m**
maximum held-ball height, **0.28 s** hop flight and four-foot landing.

The first static call took 13.0348 s including 5.7525 s loading; warm static
generation took 5.9090–6.1307 s. Peak allocated CUDA memory was 4,471,898,112
bytes. Read the [derived evaluation summary](evaluation-summary.json), unchanged
[raw outputs](probe-summary.json), [hardware](hardware.csv),
[runtime versions](runtime.json) and [model manifest](model-manifest.json).

The source commit was
[`8ca421bd95e6241ea2f676f78a9af24449796354`](https://github.com/sanjuhs/robo-pup-decision-model/commit/8ca421bd95e6241ea2f676f78a9af24449796354).
[Provenance](provenance.json) hashes every retained raw file. Model weights and
credentials are not included in this evidence directory.

Estimated GPU cost: **$0.0832595797**, bringing the session estimate to
**$0.7900419159**. The pod was deleted and absence verified. These are not final
invoices; storage reconciliation remains pending. See the
[session cost ledger](../../COST_LEDGER.md).
