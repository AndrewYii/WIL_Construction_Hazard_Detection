#!/usr/bin/env bash
# One-click launcher: starts the live hazard dashboard (if not already
# running) and opens it in the default browser. Meant to be double-clicked
# from the "Site Safety Monitor" Desktop icon, not typed by hand.

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=8090
URL="http://localhost:$PORT"

notify() {
    notify-send "Site Safety Monitor" "$1" -i camera-web 2>/dev/null || true
}

if tmux has-session -t siteguard 2>/dev/null; then
    notify "Already running — opening dashboard"
else
    notify "Starting…"
    scripts/run_session.sh --weights runs/detect/plan_a_yolov8/weights/best.pt \
        --headless --port "$PORT" --height-zone \
        > /tmp/siteguard_launch.log 2>&1 < /dev/null
fi

for _ in $(seq 1 30); do
    if curl -s -o /dev/null --max-time 1 "$URL/events" 2>/dev/null; then
        xdg-open "$URL" >/dev/null 2>&1 &
        notify "Dashboard ready"
        exit 0
    fi
    sleep 1
done

notify "Failed to start — check /tmp/siteguard_launch.log"
exit 1
