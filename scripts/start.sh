#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PWD/.cache}"
if [[ ! -f web/dist/index.html ]]; then
  printf '%s\n' 'Browser bundle missing. Run bash scripts/setup.sh first.' >&2
  exit 1
fi
exec .venv/bin/uvicorn robopup.server:app --host 127.0.0.1 --port "${PORT:-8000}"
