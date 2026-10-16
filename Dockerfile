# syntax=docker/dockerfile:1
# The Vercel frontend can run separately; this image also serves its own bundle.
FROM node:22-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392 AS frontend
WORKDIR /build/web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -f /run/secrets/build_ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/build_ca; fi; \
    npm ci --ignore-scripts
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS arena
ARG INSTALL_VLM=0
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    MUJOCO_GL=egl \
    PORT=8000 \
    ROBOPUP_ARTIFACTS=/app/artifacts \
    XDG_CACHE_HOME=/app/.cache \
    MPLCONFIGDIR=/app/.cache/matplotlib \
    HF_HOME=/app/.cache/huggingface \
    PATH=/app/.venv/bin:$PATH
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -f /run/secrets/build_ca ]; then export PIP_CERT=/run/secrets/build_ca; fi; \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libgl1-mesa-dri libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.12.23
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
# Keep the project editable: server.py resolves bundled models relative to src/.
RUN --mount=type=secret,id=build_ca,required=false \
    if [ -f /run/secrets/build_ca ]; then \
        export SSL_CERT_FILE=/run/secrets/build_ca UV_NATIVE_TLS=true; \
    fi; \
    if [ "$INSTALL_VLM" = 1 ]; then \
        uv sync --frozen --no-dev --extra vlm; \
    else \
        uv sync --frozen --no-dev; \
    fi
COPY demo/ ./demo/
COPY scripts/seed_demo.py ./scripts/seed_demo.py
COPY --from=frontend /build/web/dist/ ./web/dist/
RUN python scripts/seed_demo.py \
    && useradd --uid 10001 --create-home arena \
    && mkdir -p /app/.cache/matplotlib /app/.cache/huggingface \
    && chmod -R a+rX /app/src /app/demo /app/web \
    && chown -R arena:arena /app/artifacts /app/.cache
USER arena
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/api/health', timeout=4).read()"
CMD ["sh", "-c", "exec uvicorn robopup.server:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
