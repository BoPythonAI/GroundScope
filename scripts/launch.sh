#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/env.sh"

GS_PYTHON="${GS_PYTHON:-$GS_ROOT/envs/runtime/bin/python}"
GS_PORT="${GS_PORT:-8000}"
GS_PID_FILE="$GS_ROOT/logs/groundscope.pid"
GS_LOG_FILE="$GS_ROOT/logs/groundscope.log"
mkdir -p "$GS_ROOT/logs"

if [[ -f "$GS_PID_FILE" ]] && kill -0 "$(cat "$GS_PID_FILE")" 2>/dev/null; then
  echo "GroundScope is already running as PID $(cat "$GS_PID_FILE")"
  exit 0
fi

cd "$PROJECT_DIR"
nohup "$GS_PYTHON" -m uvicorn groundscope.api.main:app --host 0.0.0.0 --port "$GS_PORT" \
  >"$GS_LOG_FILE" 2>&1 &
echo $! >"$GS_PID_FILE"
sleep 2
if ! kill -0 "$(cat "$GS_PID_FILE")" 2>/dev/null; then
  echo "GroundScope failed to start; inspect $GS_LOG_FILE" >&2
  exit 1
fi
echo "GroundScope listening on http://0.0.0.0:$GS_PORT (PID $(cat "$GS_PID_FILE"))"
