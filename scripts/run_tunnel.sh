#!/usr/bin/env bash
# Expose the live dashboard (started by run_session.sh) to the internet via
# a Cloudflare quick tunnel, in a tmux session that survives SSH disconnects.
#
# Use this when the Spark is on a network that blocks Tailscale (e.g.
# Student Village's Fortinet firewall) — the tunnel rides over plain HTTPS,
# which Fortinet doesn't block.
#
# tmux basics:
#   tmux attach -t cftunnel      re-enter the running session
#   Ctrl+b then d                detach (leave it running)
#   scripts/stop_tunnel.sh       stop cleanly
#
# Usage: scripts/run_tunnel.sh [port]
# Defaults to port 8090 (matching run_session.sh's default).
#
# NOTE: this is a free "quick tunnel" — no Cloudflare account needed, but
# the public URL changes every time this script (re)starts. For a
# permanent URL, set up a named tunnel bound to a domain you own instead.

set -euo pipefail
cd "$(dirname "$0")/.."

SESSION=cftunnel
PORT="${1:-8090}"
LOGDIR=logs
mkdir -p "$LOGDIR"
LOG="$LOGDIR/tunnel_$(date +%Y%m%d_%H%M%S).log"

CLOUDFLARED="$HOME/bin/cloudflared"
[ -x "$CLOUDFLARED" ] || CLOUDFLARED=cloudflared

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' is already running — attaching (Ctrl+b d to detach)."
    exec tmux attach -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" \
    "$CLOUDFLARED tunnel --url http://localhost:$PORT 2>&1 | tee '$LOG'"

echo "Started tmux session '$SESSION'."
echo "  log:     $LOG"
echo "  attach:  tmux attach -t $SESSION"
echo -n "  waiting for public URL..."

URL=""
for _ in $(seq 1 20); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" 2>/dev/null | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
    echo -n "."
done
echo

if [ -n "$URL" ]; then
    echo "  public URL: $URL"
else
    echo "  URL not printed yet — check: tmux attach -t $SESSION"
fi
