#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.cache/uv}"
export npm_config_cache="${npm_config_cache:-$PWD/.cache/npm}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
if ! command -v uv >/dev/null 2>&1; then
  python3 -m venv /workspace/.robopup-tools
  /workspace/.robopup-tools/bin/pip install 'uv==0.12.23'
  export PATH="/workspace/.robopup-tools/bin:$PATH"
fi
extra_args=(--extra dev)
if [[ "${ROBOPUP_MJX:-0}" == 1 ]]; then extra_args+=(--extra mjx); fi
uv sync --frozen "${extra_args[@]}"
python3 scripts/seed_demo.py
if [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 && -w /workspace ]]; then
  python3 scripts/install_runpodctl.py
else
  printf '%s\n' 'Skipping optional cloud operator CLI on this platform.'
fi
(cd web && npm ci --ignore-scripts && npm run build)
.venv/bin/pytest -q
printf '%s\n' 'Setup validated. Start with: bash scripts/start.sh'
