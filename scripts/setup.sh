#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export npm_config_cache="${npm_config_cache:-/workspace/.cache/npm}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/workspace/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv /workspace/.robopup-tools
  /workspace/.robopup-tools/bin/pip install 'uv==0.12.23'
  export PATH="/workspace/.robopup-tools/bin:$PATH"
fi
uv sync --frozen --extra dev
python3 - <<'PY'
from pathlib import Path
import hashlib
import shutil
source = Path('demo/curious-v2')
destination = Path('artifacts/baseline/curious-v2')
if source.exists() and not destination.exists():
    expected = (source/'SHA256SUMS').read_text().split()[0]
    assert hashlib.sha256((source/'policy.zip').read_bytes()).hexdigest() == expected
    shutil.copytree(source, destination)
PY
python3 scripts/install_runpodctl.py
(cd web && npm ci --ignore-scripts && npm run build)
.venv/bin/pytest -q
printf '%s\n' 'Setup validated. Start with: bash scripts/start.sh'
