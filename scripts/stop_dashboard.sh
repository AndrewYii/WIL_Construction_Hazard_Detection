#!/usr/bin/env bash
# One-click stopper: the counterpart to start_dashboard.sh. Meant to be
# double-clicked from the "Stop Site Safety Monitor" Desktop icon.

set -uo pipefail
cd "$(dirname "$0")/.."

notify() {
    notify-send "Site Safety Monitor" "$1" -i camera-web 2>/dev/null || true
}

if ! tmux has-session -t siteguard 2>/dev/null; then
    notify "Not running"
    exit 0
fi

notify "Stopping…"
scripts/stop_session.sh > /tmp/siteguard_stop.log 2>&1 < /dev/null
notify "Stopped"
