#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/env.sh"
GS_PID_FILE="$GS_ROOT/logs/groundscope.pid"
if [[ ! -f "$GS_PID_FILE" ]]; then
  echo "GroundScope is not running."
  exit 0
fi
GS_PID="$(cat "$GS_PID_FILE")"
if kill -0 "$GS_PID" 2>/dev/null; then
  kill "$GS_PID"
  echo "Stopped GroundScope PID $GS_PID"
else
  echo "Removed stale PID file for $GS_PID"
fi
rm -f -- "$GS_PID_FILE"
