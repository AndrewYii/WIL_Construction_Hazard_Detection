#!/usr/bin/env bash
# Stop the live monitor tmux session started by run_session.sh.

set -uo pipefail
cd "$(dirname "$0")/.."

SESSION=siteguard

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "No '$SESSION' session running."
    exit 0
fi

# Ctrl+C the process, give it a moment to print its summary, then kill the pane
tmux send-keys -t "$SESSION" C-c
sleep 2
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Session '$SESSION' stopped."
echo "  events log: logs/events.jsonl"
LATEST=$(ls -t logs/session_*.log 2>/dev/null | head -1 || true)
[ -n "${LATEST:-}" ] && echo "  session log: $LATEST"
