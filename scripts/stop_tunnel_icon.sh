#!/usr/bin/env bash
# Stops the permanent Cloudflare Tunnel systemd --user service. Rarely
# needed day-to-day (it's meant to run continuously) — mainly for
# troubleshooting, or to temporarily take the dashboard off the public
# internet. Meant to be double-clicked from a Desktop icon.
#
# It restarts automatically on the next login/boot (systemd, enabled +
# lingering) unless you also run: systemctl --user disable cloudflared

set -uo pipefail
cd "$(dirname "$0")/.."

notify() {
    notify-send "Remote Access" "$1" -i network-vpn 2>/dev/null || true
}

if ! systemctl --user is-active --quiet cloudflared; then
    notify "Already stopped"
    exit 0
fi

notify "Stopping tunnel…"
systemctl --user stop cloudflared
notify "Stopped (will restart automatically at next login/boot)"
