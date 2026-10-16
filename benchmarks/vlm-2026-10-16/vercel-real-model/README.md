# Actual local Vercel services calling Qwen

**Passed:** a genuine `Stop.` planning request with the recorded 384×384
robot-eye PNG traveled through the real local Vercel services to the loaded
native Qwen3-VL-2B-Instruct model and returned a validated stop plan.

The route was:

1. Vercel CLI 64.3.1 `dev -L` on `127.0.0.1:3010`.
2. HTTP gateway `/api/plan`.
3. Vercel's injected `PLANNER_URL=http://127.0.0.1:44457`, calling the private
   planner's `/plan` route. The helper did not set `PLANNER_URL`.
4. Authenticated native `/v1/chat/completions` on port 8000.
5. Actual local Qwen inference on CPU, BF16, configured for four CPU threads.

The request took **42.4086 wall seconds** after the model was already warmed by
the browser. Its remote-provider request timer was 42.1063 seconds. Complete
startup, request and cleanup took 49.6287 seconds. This is one local measurement,
not a general latency benchmark. The response was HTTP 200 and all six checks
passed. [Exact result and returned JSON](result.json).

The returned plan contained `stop` with object `none`. The planner service
returns validated plan JSON rather than raw generative text, so this record
does not invent or claim to retain the original model text. The model's summary
uses past tense; it does not prove the robot physically stopped. This request
performed **planning only**, with `physics_executed=false`.

This was **local development**, without a Vercel cloud login or hosted
deployment. The gateway and internal planner loaded no model weights. A private
ephemeral workshop key was read from a mode-0600 file into process memory and
was excluded from all evidence; the Runpod account key was not used or passed
to these processes.

The helper recorded the binding from its own Vercel descendant processes and
verified gateway health and the internal planner status through that binding.
Its owned gateway, planner, web and resource ports were closed afterward;
the native backend on port 8000 remained running.
[Independent cleanup check](cleanup-verification.json).

Retained evidence:

- [Exact helper source](probe-helper.py), including process ownership and cleanup.
- [Sanitized local CLI log](vercel-local-sanitized.log).
- [Actual camera input](eye-start.png).
- [File hashes and source provenance](provenance.json).

The source ref was
[`77bf40c1f0819cab5175621d3090d2991b850eae`](https://github.com/sanjuhs/robo-pup-decision-model/commit/77bf40c1f0819cab5175621d3090d2991b850eae);
individual service/runtime/config hashes are also captured in `result.json`.
No weights, private token values or cloud credentials are included.
