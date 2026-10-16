# Runpod cost ledger

The aggregate **estimated GPU cost is $0.7900419159**, inside the $30 ceiling.
These are creation-to-deletion estimates, not final invoices. Storage charges
have not been reconciled. All six observed session pods were deleted and their
absence was verified; no persistent network volumes were created.

| Observed pod experiment | Estimated GPU USD | Lifecycle evidence |
| --- | ---: | --- |
| Original GPU physics interface failure | 0.1102997778 | [Attempt 1](runpod-attempt-1/lifecycle.json) |
| Original GPU physics/PPO retry | 0.1666363611 | [Attempt 2](runpod-attempt-2/lifecycle.json) |
| Rejected v3 arena planner | 0.1975554861 | [v3](gpu-planner-v3/lifecycle.json) |
| v4 bootstrap network failure | 0.1032827672 | [v4 attempt 1](gpu-planner-v4/attempt1-network-failure/lifecycle.json) |
| v4 arena planner training | 0.1290079440 | [v4 attempt 2](gpu-planner-v4/attempt2/lifecycle.json) |
| Qwen VLM inference on RTX 4090 | 0.0832595797 | [VLM GPU](vlm-2026-10-16/gpu-4090/lifecycle.json) |
| **Total** | **0.7900419159** | [Machine-readable ledger](gpu-planner-v4/cost-ledger.json) |

The VLM pod reported **$0.89/hour** and was monitored for **336.7803 seconds**.
Its account key was not passed into the pod. The exported artifact checksum,
deletion confirmation and post-deletion absence are retained in its lifecycle
record. This was pretrained-model inference, not model training; the closed-loop
mission used native MuJoCo on CPU alongside GPU VLM inference.

Two earlier RTX 5090 allocation requests returned HTTP 500 and HTTP 400 without
an observed pod. They contribute **zero observed allocated GPU time** to this
ledger. This does not turn a failed API request into a billing receipt:

- [Attempt 1 raw record](vlm-2026-10-16/gpu-allocation-attempt1/raw-lifecycle.json)
  is unchanged. Its original elapsed-monitor-time cost arithmetic is explicitly
  corrected by the [independent zero-pod reconciliation](vlm-2026-10-16/gpu-allocation-attempt1/reconciliation.json).
- [Attempt 2](vlm-2026-10-16/gpu-allocation-attempt2/lifecycle.json) retains the
  raw uncertainty flag and separate $0.625 planned-bound upper estimate. The
  subsequent successful VLM allocation recorded zero pre-existing pods.

The older `followup_gpu_estimate_usd` field in the JSON ledger remains the
historical v3/v4 planner subtotal, $0.4298461973. VLM inference is itemized
separately; the session total includes both.
