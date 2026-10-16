# Run the arena in a browser

The prepared Vercel deployment has three services: a Vite browser frontend,
a lightweight public FastAPI gateway, and an internal FastAPI vision-planning
service. The live MuJoCo simulation remains a separate long-lived backend.
Neither Vercel Python service imports MuJoCo, Torch, or local model weights.
These files have not been deployed; the final service names, public routes and
internal binding are prepared for the user's requested review.

Current Vercel documentation supports WebSockets in public beta, including
Python, on Fluid Compute. Connections still close at the function's maximum
duration, and memory is instance-local. Normal Python bundles are limited to
500 MB; larger-function support is a separate beta. This project does not
package its native simulation, multi-GB local model and persistent arena sessions
into that runtime. See the current [WebSocket guide](https://vercel.com/kb/guide/do-vercel-serverless-functions-support-websocket-connections)
and [function limits](https://vercel.com/docs/functions/limitations).

## CPU backend container

From the repository root:

```bash
docker build -t robopup-arena:cpu .
docker run --rm -p 127.0.0.1:8000:8000 robopup-arena:cpu
```

Open `http://localhost:8000`. The image builds the browser bundle, installs the
frozen Python lock with `uv`, includes the bundled policies/planner, seeds demo
experiments, and runs as user 10001 with EGL/Mesa rendering. This default image
uses the compact trained arena planner; it does not download or enable a large
vision-language model. Check `http://localhost:8000/api/health` for availability.

If a managed network uses a private HTTPS proxy certificate, pass the trusted
CA bundle as a BuildKit secret rather than disabling TLS verification:

```bash
docker build --secret id=build_ca,src=/path/to/trusted-ca-bundle.crt \
  -t robopup-arena:cpu .
```

The optional CA file is mounted only during dependency installation and is not
copied into the resulting image. A runtime inference provider behind the same
proxy also needs an explicitly trusted runtime CA bundle.

The server binds `0.0.0.0` inside the container and honors `PORT` (8000 by
default). The example publishes only localhost. For a remotely hosted backend,
use an HTTPS reverse proxy with WebSocket upgrades, sufficiently long WebSocket
timeouts, and a reachable `wss://` endpoint. Use a single Uvicorn worker: the
current session capacity counter and model cache are process-local.

Persist experiments by mounting a writable volume at `/app/artifacts`; the
volume must permit writes by UID 10001. An empty volume hides the image's seeded
demo comparison runs, although the directly bundled default policy still loads.
The current arena has no durable session reconnection or multi-node scheduler.

## Enable a vision-language model

Build optional inference dependencies:

```bash
docker build --build-arg INSTALL_VLM=1 -t robopup-arena:vlm .
```

The local model option is `Qwen/Qwen3-VL-2B-Instruct`, a pretrained 2B-class
vision-language model. Its weights are downloaded separately rather than baked
into this image. This public model does not require a Hugging Face token; a
token can help with Hub rate limits. Never place a token in a Docker build
argument, a GitHub file, or a `VITE_` variable.

With the optional local Python dependencies installed, download a local copy:

```bash
uv run --extra vlm python scripts/download_vlm.py \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203
```

Start with that model directory mounted read-only:

```bash
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$(pwd)/artifacts/vlm:/models:ro" \
  -e ROBOPUP_VLM_MODEL_DIR=/models/Qwen3-VL-2B-Instruct \
  -e ROBOPUP_DEFAULT_BRAIN=vlm \
  robopup-arena:vlm
```

Allow several GB of disk and memory for weights and inference. CPU inference
can be slow; no production concurrency or interactive latency is promised by
the image. The CPU Torch lock does not become GPU-enabled by adding `--gpus`.
A GPU-specific runtime requires a tested CUDA Torch installation and a measured
host; this Dockerfile does not provision or benchmark one.

Alternatively, the Python server can use an OpenAI-compatible vision endpoint:
set `ROBOPUP_VLM_BASE_URL`, `ROBOPUP_VLM_MODEL`, and `ROBOPUP_VLM_API_KEY`
as backend runtime variables. A configured base URL selects remote inference;
otherwise the optional runtime uses the explicitly downloaded local checkpoint.
That provider's usage is billed under its own account, not the static Vercel
site. Do not expose its key to the browser. Check `/api/vlm` for provider status;
a configured provider is not proof of a successful multimodal inference.

## Review the Vercel services and routes

The root [vercel.json](../vercel.json) uses the current
[Services configuration](https://vercel.com/docs/services), rather than a single
catch-all route to the original heavyweight Python project.

| Service | Root | Framework | Public traffic | Responsibility |
|---|---|---|---|---|
| `app` | `services/app` | FastAPI | `/api/:path*` | HTTP health/config, planning gateway, optional experiments/audio proxy |
| `web` | `web` | Vite | `/:path*`, after the API rule | Static browser UI and assets |
| `planner` | `services/planner` | FastAPI | None | Validate image/instruction, call remote VLM, validate returned mission |

The first matching public rewrite owns the request. `/api/health` reaches
`app`, while `/`, frontend assets and other paths reach `web`. The `planner`
service has no public rewrite. These service roots have independent minimal
requirements, preventing the root project's Torch/MuJoCo dependencies from
being bundled into the gateway or planner functions.

`app` declares a caller-side internal binding to `planner` with type `service`,
format `url`, and environment name `PLANNER_URL`. Vercel injects that URL into
the gateway's functions at runtime. `/api/plan` reads it when handling a request
and calls the internal `/plan` route. The URL is neither hardcoded nor exposed
in the frontend. Runtime bindings do not resolve during a Vite build or in
middleware; static browser code never reads `PLANNER_URL`. See
[service routing](https://vercel.com/docs/services/routing) and
[bindings](https://vercel.com/docs/services/bindings).

The internal planner's install step copies the exact repository
`vlm_protocol.py` and lightweight `vlm_runtime.py` modules into its isolated
bundle, rather than maintaining a second action schema. The remote provider
gets the supplied PNG pixels, instruction, energy and held-object bit. Returned
plans contain at most eight supported action/object pairs. This HTTP API returns
a plan; it does not move the robot or establish physical task completion.
The interactive arena's mission execution still lives on the external simulation
backend, with its own VLM runtime and sequential physical completion checks.

Planning request format for `POST /api/plan`, with an HTTP bearer workshop key:

```json
{
  "instruction": "Inspect the scene, then fetch the ball",
  "image_png": "<base64 PNG from the current robot-eye camera>",
  "energy": 0.8,
  "held": false
}
```

No configured provider produces a clear 503 response. Invalid images, missing
access keys, provider failures and invalid mission JSON are rejected. A canned
answer or compact planner is never substituted for a missing remote VLM.

## Connect the services to the simulation backend

After reviewing the service names, routes and binding:

1. Import the repository into the intended personal Vercel account with the
   project root at the repository root. Each service builds from its own root.
2. Set public build variable
   `VITE_ARENA_BACKEND_URL=https://your-physics-backend.example`. The URL is
   intentionally included in the browser bundle and serves `/api/*` and `/ws`
   directly through HTTPS/WSS. Optionally set backend runtime variable
   `ROBOPUP_BACKEND_URL` on Vercel so the HTTP gateway can proxy experiments
   and transcription; simulation WebSockets still connect directly to the
   external backend.
3. Set Vercel runtime secrets `ROBOPUP_ARENA_TOKEN`,
   `ROBOPUP_VLM_BASE_URL`, `ROBOPUP_VLM_MODEL`, and `ROBOPUP_VLM_API_KEY`
   for the gateway/internal planner. The model endpoint must support image input
   through the OpenAI-compatible chat completions API. Vercel generates
   `PLANNER_URL`; do not set it as a build variable or public URL.
4. On the simulation backend, set `ROBOPUP_ALLOWED_ORIGINS` to the exact
   frontend origin and set `ROBOPUP_ARENA_TOKEN` through host secrets. Configure
   that backend's own local or remote VLM runtime for interactive mission
   execution; the separate Vercel planning HTTP API is not automatically the
   simulation's inference provider.
5. In **Backend connection**, enter the workshop access key. It is kept in
   browser session storage, sent as an HTTP bearer token and as the first
   WebSocket authentication message. It is not embedded in the public bundle or
   put into a URL. This is a shared workshop gate, not per-user accounts/billing.

The backend URL can also be entered through **Backend connection** and saved
locally, so a static site can be connected without rebuilding it. An HTTPS
Vercel page cannot connect to a plain HTTP backend. Microphone capture requires
a secure browser context and the user's microphone permission.

To deploy after the requested configuration review, use an authenticated local
Vercel CLI from the repository root and review the account/project it links.
Use `vercel --prod` only for the intended account/project. Preview deployments have distinct origins: add
the exact preview origin to the backend allowlist when needed rather than
disabling origin checks. Never put API keys in command history or commit a
`.vercel` account/project directory.

## Verify the end-to-end deployment

Open the deployed UI and confirm all of the following:

- `/api/health` is reachable through the selected backend and `/ws` opens with
  valid workshop authentication. The robot moves only while physics advances.
- The first-person camera changes when the physical robot turns or moves.
- A VLM command produces a real provider response and a validated sequence of
  supported actions. Provider/configuration failure must be visible; the
  compact planner must not be presented as a large language model.
- Action completion is checked in the simulation. A model's statement that it
  fetched an object does not count as physical completion.
- Invalid workshop keys and unapproved origins cannot open a simulation or use
  protected HTTP operations. Do not share private inference-provider keys with
  participants.

Validation performed for these files: official Vercel schema instance validation
and eight local service tests, including real HTTP calls through the gateway,
internal planner and a clearly labeled fake provider fixture. That fixture
verifies transport, exact camera image forwarding and mission validation; it
does not prove actual VLM inference. A subprocess confirms the planner bundle
imports no Torch/MuJoCo/Transformers, including in a separate minimal-dependency
virtual environment with hash-verified installation.

The real Vercel CLI (`64.3.1`) also ran `vercel dev -L` locally without a cloud
login. `/` returned Vite HTML, `/api/health` returned the gateway with a configured
generated binding, and `/api/vlm` successfully called the internal planner's
`/status` route. Public `/status` returned frontend HTML rather than exposing
the internal service. These local processes were then stopped; this is not a
hosted deployment or a test of a deployed private network.

A subsequent [genuine-model service probe](../benchmarks/vlm-2026-10-16/vercel-real-model/)
sent an actual 384×384 eye PNG and “Stop” through local `/api/plan`, the
independently observed Vercel-injected `PLANNER_URL`, and the authenticated
native `/v1/chat/completions` endpoint. The warm CPU Qwen model returned a valid
stop plan in **42.41 request wall seconds**. All six checks passed; all seven
owned helper ports were closed afterward. The returned plan did not execute
physics. This verifies the local service-to-model chain, while production Vercel
authentication, hosted routing and a public backend remain unverified.

The initial container build required a
proxy CA trust correction; its first non-root runtime check exposed restrictive
checkout file permissions, addressed by making only bundled source/models/web
assets readable. Another build exhausted workspace disk while retaining old
installer layers; those disposable images/caches were removed and uv's image
cache disabled. The corrected 1.41 GB CPU image
`sha256:253f183d89bf41fc50fd73918c4745d2819db46ff71cc2f6d80b198b6eb3329d`
passed non-root imports, actual 384×384 EGL eye rendering, health and frontend
HTTP responses, and an authenticated WebSocket with physics time advancing from
0.04 to 0.08 seconds using `PORT=8056`. Its disposable runtime container was
then stopped and removed. The optional VLM-extra image was not built or tested.

Vercel publication, real provider inference, container behavior and deployed
physical behavior are separate validation steps. No hosted deployment or deployed
service binding has been verified by these local checks.
