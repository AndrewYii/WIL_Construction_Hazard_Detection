# DGX Spark setup — hazard detection stack

How to run the full stack against the NVIDIA DGX Spark: Ollama serving the
Phase 2 LLMs/VLMs on port 11434, and the real-time detector (`app/live.py`)
either on the Spark itself (recommended) or on a laptop pointing at it.

## 1. Ollama on the Spark

Expose Ollama to the LAN (default is localhost only):

```bash
sudo systemctl edit ollama
# add these lines in the editor:
#   [Service]
#   Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl restart ollama
```

Pull the models (order matters only for disk planning — the code picks the
first available from each chain):

```bash
# Phase 2 report writer chain
ollama pull gpt-oss:120b      # ~65 GB — main report model (MoE, fast per token)
ollama pull gemma4:31b        # ~20 GB — strong fallback (dense, Apr 2026)
ollama pull qwen3:32b         # ~20 GB — fast fallback

# VLM chain (Plan D, frame understanding)
ollama pull gemma4:26b        # ~18 GB — main VLM (MoE, ~4B active params, fast + vision)
ollama pull qwen2.5vl:32b     # ~21 GB — fallback / comparison
ollama pull llava:7b          # ~4.7 GB — last resort
```

Verify from any machine on the network:

```bash
curl http://<spark-ip>:11434/api/tags
```

## 2. Environment variables (any machine running this repo)

```bash
export OLLAMA_HOST=http://<spark-ip>:11434    # Windows: set OLLAMA_HOST=...
```

Optional overrides (defaults in `app/config.py`): `REPORT_MODELS`,
`VLM_MODELS` (comma-separated chains), `ALERT_COOLDOWN_SEC`,
`PROXIMITY_DISTANCE_RATIO`, `DASHBOARD_PORT`.

Check connectivity: `python app/llm_client.py` prints the server status and
which models the chains resolved to.

## 2b. Everything on the Spark (single-server mode)

The whole stack runs on the Spark itself; phones/laptops only need a browser.

```bash
git clone <repo> && cd WIL_Construction_Hazard_Detection
git checkout Spark
python3 -m venv venv && source venv/bin/activate
pip install -r app/requirements.txt

# the Spark is ARM64 (GB10) — install the CUDA build of torch explicitly:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.cuda.is_available())"   # must print True

sudo apt install espeak-ng    # voice clips for server-side audio
```

Then:
- live monitor + dashboard: `scripts/run_session.sh` → phone opens
  `http://<spark-ip>:8090` (tap the voice button once — mobile browsers
  need one tap before they allow speech)
- upload app (optional): `streamlit run app/app.py --server.address 0.0.0.0`
  → `http://<spark-ip>:8501`
- `OLLAMA_HOST` can stay default on the Spark itself (localhost:11434)

## 3. Real-time monitor

```bash
# on the Spark (headless — a browser anywhere on the LAN is the monitor)
python app/live.py --weights runs/detect/plan_a_yolov8/weights/best.pt \
    --source 0 --headless --port 8090

# rehearsal with no GPU / camera / weights
python app/live.py --video some_clip.mp4 --mock --headless

# with a local display instead
python app/live.py --weights ... --source rtsp://camera-url
```

Open `http://<spark-ip>:8090` — live annotated stream, hazard banner,
per-use-case counters, alert log, and browser-side voice alerts (Web Speech
API), so the announcement plays wherever the dashboard is open even though
the Spark has no speaker. Server-side audio (speaker plugged into the
detection machine) plays the same pre-synthesized clips; disable with
`--no-audio`.

For unattended operation use `scripts/run_session.sh` (tmux) and
`scripts/stop_session.sh`.

## 3b. Site testing: cameras

**Phone as camera (easiest field test).** Install an IP camera app on the
phone (Android: "IP Webcam"), start its server, note the URL it shows, then
on the Spark:

```bash
python app/live.py --weights ... --headless \
    --source http://<phone-ip>:8080/video
```

Walk the site with the phone; watch the dashboard (and hear alerts) from any
browser. Phone and Spark must be on the same network (or both on Tailscale).

**Intel RealSense.** Must be plugged into the detection machine by USB3 —
depth does not survive ordinary network streaming. Its RGB stream registers
as a normal webcam, so `--source 0` (or 1) works with no code changes.
Planned upgrade once mounted: read aligned depth via `pyrealsense2` and
convert the proximity hazard from pixel distance to true metres — a large
accuracy gain for the proximity use case.

## 4. Hazard use cases

| Use case | Trigger | Spoken alert |
|---|---|---|
| Proximity | worker box overlaps / within 0.25 frame-diagonal of a vehicle box | "Warning! Worker too close to vehicle. Move away now." |
| Vehicle | vehicle centroid moving > 4%/s of frame diagonal while workers present | "Caution! Heavy vehicle moving in work zone." |
| Height (experimental) | worker box in the upper zone of the frame (`--height-zone`) | "Warning! Worker working at height near edge." |

Alerts debounce (3 consecutive detection passes) and cool down (6 s per
type). Every fired alert is appended to `logs/events.jsonl`.

Height is a camera-geometry heuristic and off by default — real height
detection waits on the planned synthetic data (see project context). This
matches industry experience that height is the hardest hazard to model.

## 5. Performance / architecture notes

- `scripts/benchmark_spark.py --append-readme` produces the FPS table
  (yolov8n..x, yolo11n/s/m, and the fine-tuned best.pt) for the report.
- Upgrade path when more speed is needed on the Spark:
  1. Export the fine-tuned model to TensorRT:
     `yolo export model=best.pt format=engine half=True` and pass the
     `.engine` file to `--weights` — typically 2-3x faster than PyTorch.
  2. Retrain on YOLO11m — same dataset and pipeline, better accuracy/speed
     tradeoff than YOLOv8m.
  3. Keep `--imgsz 640` for accuracy; drop to 480 only if the camera is
     close to the action.

## 6. Phase 2 reports

- Streamlit upload app: report generated automatically after processing.
- Live dashboard: "Generate report" button (`GET /report`) builds the report
  from the running session's statistics via the Spark LLM chain.
- If the Spark is unreachable, both fall back to a built-in template — the
  pipeline never crashes because of the LLM.
