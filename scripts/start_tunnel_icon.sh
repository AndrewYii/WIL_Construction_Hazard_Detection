#!/usr/bin/env bash
# One-click "show my remote link" — the permanent Cloudflare Tunnel
# (systemd --user service `cloudflared`, see ~/.config/systemd/user/cloudflared.service)
# runs continuously in the background and does not need to be started by
# this script. This just checks everything is healthy and shows the fixed
# URL in a copyable dialog. Meant to be double-clicked from a Desktop icon.
#
# Use this when Tailscale is offline (Student Village's Fortinet firewall
# blocking it — see the Obsidian vault, Deployment Runbook Phase 4B).
# Requires the dashboard to already be running (start the "Site Safety
# Monitor" icon first).

set -uo pipefail
cd "$(dirname "$0")/.."

PORT=8090
URL="https://wil.andrewtest.me"

notify() {
    notify-send "Remote Access" "$1" -i network-vpn 2>/dev/null || true
}

if ! curl -s -o /dev/null --max-time 1 "http://localhost:$PORT/events" 2>/dev/null; then
    notify "Dashboard isn't running yet — start 'Site Safety Monitor' first"
    exit 1
fi

if ! systemctl --user is-active --quiet cloudflared; then
    notify "Tunnel service isn't running — starting it…"
    systemctl --user start cloudflared
    sleep 3
fi

# The local/campus DNS resolver can lag behind Cloudflare's own DNS for a
# freshly-created record (stale negative cache) — query Cloudflare directly
# (1.1.1.1) so this check isn't fooled by that, since the tunnel itself is
# reachable to real-world users the moment Cloudflare's own DNS is right.
HOST="${URL#https://}"
IP=$(dig @1.1.1.1 +short "$HOST" 2>/dev/null | head -1)
if [ -n "$IP" ]; then
    HEALTH_CHECK="curl -s -o /dev/null --max-time 8 --resolve $HOST:443:$IP $URL"
else
    HEALTH_CHECK="curl -s -o /dev/null --max-time 8 $URL"
fi

if ! $HEALTH_CHECK; then
    notify "Tunnel isn't responding yet — try again in a few seconds"
    exit 1
fi

notify "Remote link ready"
zenity --entry --title="Remote Access — Site Safety Monitor" \
    --text="Share this link (permanent — always the same, works from any network):" \
    --entry-text="$URL" --width=500 2>/dev/null || true
