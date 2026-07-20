# Headless Spark boot sanity check

Verify the Spark boots and is reachable with no monitor attached.

1. Power on and wait ~60-90 s for boot.
2. From a laptop on the same network:
   ```bash
   ssh <user>@<spark-ip>
   ```
   If you don't know the IP yet, attach a monitor once and run `ip addr`
   (look for the LAN interface, e.g. `enp*` or `wlan*`), then **record the
   IP** — better, give the Spark a static lease in the router so it never
   changes.
3. Once in via SSH, confirm services:
   ```bash
   systemctl status ollama          # LLM server
   curl -s localhost:11434/api/tags # models respond
   nvidia-smi                       # GPU visible
   ```
4. Fallbacks if SSH fails:
   - Remote desktop: `gnome-remote-desktop` (enable once in Settings →
     Sharing while a monitor is attached).
   - Last resort: reattach monitor + keyboard.

Live monitor in headless operation: `scripts/run_session.sh` starts
`app/live.py --headless` inside tmux (survives SSH disconnect); the
dashboard at `http://<spark-ip>:8090` is the monitor.
