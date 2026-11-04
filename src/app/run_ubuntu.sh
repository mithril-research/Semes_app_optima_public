#!/usr/bin/env bash
# Ubuntu server: bash src/app/run_ubuntu.sh setup [cpu|gpu]
# Start: bash src/app/run_ubuntu.sh [start]
# Auxiliary models: src/보조모델/{스텝시간,실린더}/DSVn first; repository-level copy second.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
VENV="${OPTIMA_VENV:-$HERE/venv}"
ACTION="${1:-start}"

case "$ACTION" in
  setup-offline)
    [ -n "${2:-}" ] || { echo "Usage: $0 setup-offline /path/to/packages" >&2; exit 2; }
    exec "${PYTHON:-python3}" "$HERE/ubuntu_offline.py" install "$2" --venv "$VENV"
    ;;
  setup)
    MODE="${2:-cpu}"
    case "$MODE" in cpu|gpu) ;; *) echo "Usage: $0 setup [cpu|gpu]" >&2; exit 2 ;; esac
    PYTHON="${PYTHON:-python3}"
    "$PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python 3.10–3.12 (Ubuntu 22.04/24.04)"'
    "$PYTHON" -m venv "$VENV"
    PY="$VENV/bin/python"
    "$PY" -m pip install --upgrade pip
    if [ "$MODE" = cpu ]; then
      "$PY" -m pip install 'torch>=2.4,<2.7' --index-url https://download.pytorch.org/whl/cpu
      "$PY" -m pip uninstall -y onnxruntime-gpu
      "$PY" -m pip install -r "$HERE/requirements-ubuntu-cpu.txt"
    else
      "$PY" -m pip install 'torch>=2.7,<2.8' --index-url https://download.pytorch.org/whl/cu128
      "$PY" -m pip uninstall -y onnxruntime
      "$PY" -m pip install -r "$HERE/requirements-ubuntu.txt"
    fi
    "$PY" -m pip check
    echo "Installed ($MODE). Start: bash $0"
    exit 0
    ;;
  start) ;;
  *) echo "Usage: $0 [setup [cpu|gpu]|setup-offline /path/to/packages|start]" >&2; exit 2 ;;
esac

[ -x "$VENV/bin/python" ] || { echo "Run: bash $0 setup" >&2; exit 1; }
cd "$ROOT"
export IO_HOST="${IO_HOST:-0.0.0.0}"
export PORT="${PORT:-8000}"
export IO_KEEP_ALIVE=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
# One process owns the in-memory jobs, file watcher and inspection engine.
echo "OPTIMA: http://<server-ip>:$PORT (bind $IO_HOST)"
exec "$VENV/bin/python" -m src.app.server
