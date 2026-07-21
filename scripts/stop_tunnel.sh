#!/usr/bin/env bash
# Stop the Cloudflare tunnel started by run_tunnel.sh.

set -uo pipefail
cd "$(dirname "$0")/.."

SESSION=cftunnel

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "No '$SESSION' session running."
    exit 0
fi

tmux send-keys -t "$SESSION" C-c
sleep 2
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Session '$SESSION' stopped."
