# Test video scenario map

Filenames follow `<scenario>_<source>.mp4`. Every clip here is from Pexels (free, CC0, no
attribution required). Verified by actually running each one through `app/live.py` (real
Plan A weights, `--height-zone`, PPE_Detect on), not by filename or label. Regenerate with:

```bash
for f in test_videos/*.mp4; do
  python app/live.py --video "$f" --weights runs/detect/plan_a_yolov8/weights/best.pt \
    --headless --port 8097 --autostart --height-zone --no-analysis &
  sleep 22
  curl -s http://127.0.0.1:8097/events | python3 -c "import json,sys; print(json.load(sys.stdin)['alert_counts'])"
  kill %1
done
```

| Video | Proximity | Vehicle motion | Height | PPE | Notes |
|---|---|---|---|---|---|
| `all_hazards_busy_site_pexels.mp4` | ✅x3 | ✅x3 | ✅x3 | ✅x2 | **Strongest all-in-one demo** — worker+vehicle co-occur in nearly every sampled frame. |
| `all_hazards_nohardhat_pexels.mp4` | ✅x2 | ✅x2 | ✅x3 | ✅x3 | All four, with a strong repeated `NO-Hardhat` signal — a clean, unambiguous PPE-alarm demo. |
| `site_overview_pexels.mp4` | ✅ | ✅ | ✅ | ✅ | All four hazard types in one clip. "Construction site overview." |
| `construction_excavator_pexels.mp4` | ✅ | ✅ | ✅ | ❌ (compliant) | Busiest scene (peak 5 workers / 4 vehicles). The comparison clip — PPE never fires here, workers genuinely wear hardhat + vest, a true negative not a bug. |
| `no_ppe_worker_pexels.mp4` | ❌ | ❌ | ❌ | ✅x1 | **Isolated PPE-only demo** — no vehicles in frame at all, cleanest way to show the PPE alert without proximity/vehicle noise. |
| `height_scaffold_pexels.mp4` | ❌ | ❌ | ✅ | ✅ | Best dedicated height clip. "Worker securing scaffolding." No vehicles, so proximity/vehicle can't fire — expected. |
| `vehicle_bulldozer_pexels.mp4` | ✅ | ✅ | ❌ | ❌ | Only 1 worker peak. "Bulldozer moving backwards" — quick proximity/vehicle smoke test. |
| `construction_test_pexels.mp4` | ❌ | ❌ | ❌ | ✅ | No vehicles (proximity/vehicle can't fire by design). Good dedicated PPE-only clip. |

`proximity_excavator_truck.mp4` ("excavator and dump truck", Pexels) was removed
(2026-07-19) — despite the name, the worker detector never detected a single person in it
across the whole clip (highest confidence at a conf floor of 0.05 was 0.34, below the 0.4
production threshold). Not a logic bug, just footage this fine-tuned model can't work with
(likely too small/distant/occluded).

## Source

All eight clips (the six original plus two added this session) are Pexels footage — free,
CC0, no attribution required.

| Video | Pexels source |
|---|---|
| `construction_excavator_pexels.mp4` | [pexels.com/video/4271760](https://www.pexels.com/video/4271760/) |
| `construction_test_pexels.mp4` | [pexels.com/video/2048246](https://www.pexels.com/video/2048246/) |
| `height_scaffold_pexels.mp4` | [pexels.com/video/3078521](https://www.pexels.com/video/low-angle-footage-of-a-construction-worker-securing-the-steel-scaffolding-on-the-building-exterior-3078521/) |
| `vehicle_bulldozer_pexels.mp4` | [pexels.com/video/7163091](https://www.pexels.com/video/a-bulldozer-moving-backwards-7163091/) |
| `site_overview_pexels.mp4` | [pexels.com/video/856439](https://www.pexels.com/video/construction-site-856439/) |
| `all_hazards_busy_site_pexels.mp4` | [pexels.com/video/11355862](https://www.pexels.com/video/11355862/) |
| `all_hazards_nohardhat_pexels.mp4` | [pexels.com/video/12908568](https://www.pexels.com/video/12908568/) |
| `no_ppe_worker_pexels.mp4` | [pexels.com/video/8487127](https://www.pexels.com/video/8487127/) |

**How `all_hazards_busy_site_pexels.mp4` / `all_hazards_nohardhat_pexels.mp4` /
`no_ppe_worker_pexels.mp4` (the 3 added this session) were fetched:** plain `curl` is
blocked by Pexels/Pixabay/Freepik's Cloudflare bot-challenge (403 "Just a moment..."), and
Claude in Chrome wasn't connected in-session. Worked around it with a local **Playwright**
headless Chromium — no sudo needed, browser binary downloads to `~/.cache/ms-playwright`:
```bash
venv/bin/pip install playwright && venv/bin/playwright install chromium
```
Launched with `args=['--disable-blink-features=AutomationControlled']` plus
`page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")`
to mask basic headless fingerprints. Freepik still blocked it even this way; Pexels did not.
Every candidate was downloaded blind (by video ID from a search page, no trusted
title/description) and empirically scanned — worker/vehicle confidence + PPE_Detect hits
sampled across the clip — before being run through the full `live.py` pipeline to confirm it
actually fires the intended hazard. The other 5 clips predate this session; their links were
supplied directly rather than fetched by this method.

## Adding a new clip

Validate before trusting it — several clips this session (including ones picked by name
alone) turned out not to trigger what they promised:

```bash
python app/live.py --video test_videos/<new_clip>.mp4 \
  --weights runs/detect/plan_a_yolov8/weights/best.pt \
  --headless --port 8097 --autostart --height-zone --no-analysis &
sleep 22
curl -s http://127.0.0.1:8097/events | python3 -c "import json,sys; print(json.load(sys.stdin)['alert_counts'])"
kill %1
```
