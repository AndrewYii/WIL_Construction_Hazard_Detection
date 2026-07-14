#!/usr/bin/env bash
# Start the live hazard monitor inside a tmux session on the Spark, so it
# survives SSH disconnects.
#
# tmux basics:
#   tmux attach -t siteguard     re-enter the running session
#   Ctrl+b then d                detach (leave it running)
#   scripts/stop_session.sh      stop cleanly
#
# Usage: scripts/run_session.sh [extra live.py args...]
# Defaults to headless mode on port 8090 with the fine-tuned weights.

set -euo pipefail
cd "$(dirname "$0")/.."

SESSION=siteguard
LOGDIR=logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/session_$(date +%Y%m%d_%H%M%S).log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' is already running — attaching (Ctrl+b d to detach)."
    exec tmux attach -t "$SESSION"
fi

PY=venv/bin/python
[ -x "$PY" ] || PY=python3

ARGS="${*:---weights runs/detect/plan_a_yolov8/weights/best.pt --headless --port 8090}"

tmux new-session -d -s "$SESSION" \
    "$PY app/live.py $ARGS 2>&1 | tee '$LOG'"

echo "Started tmux session '$SESSION'."
echo "  log:       $LOG"
echo "  events:    logs/events.jsonl"
echo "  attach:    tmux attach -t $SESSION"
echo "  dashboard: http://$(hostname -I 2>/dev/null | awk '{print $1}'):8090"
