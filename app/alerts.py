"""
Production alert engine for live hazard detection.

Per hazard type (proximity / vehicle / height):
  - debounce: hazard must persist ALERT_TRIGGER_FRAMES consecutive detection
    passes before the alarm fires (kills single-frame flickers)
  - cooldown: ALERT_COOLDOWN_SEC between repeats of the same alert so the
    voice does not spam the site
  - every fired alert is appended to logs/events.jsonl (audit trail)

Audio design: spoken WAV clips are synthesized ONCE at startup with pyttsx3
(Windows SAPI / Linux espeak-ng) into assets/audio/, then playback is a cheap
non-blocking call on a daemon thread — the detection loop is never blocked.
If TTS or a sound device is unavailable the engine still runs (events log +
dashboard voice via the browser still work); it just plays a beep or nothing.
"""

import json
import platform
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import config

# The four hazard use cases (see docs/SPARK_SETUP.md).
ALERT_PHRASES = {
    "proximity": "Warning! Worker too close to vehicle. Move away now.",
    "vehicle": "Caution! Heavy vehicle moving in work zone.",
    "height": "Warning! Worker working at height near edge.",
    "ppe": "Warning! Worker without required protective equipment detected.",
}


# --------------------------------------------------------------------------
# Audio backend
# --------------------------------------------------------------------------

def synthesize_clips(audio_dir: Path = config.AUDIO_DIR) -> dict[str, Path]:
    """Pre-generate one WAV per phrase. Returns {hazard_type: wav_path} for
    the clips that exist afterwards. Safe to call every startup — skips
    files already on disk."""
    audio_dir.mkdir(parents=True, exist_ok=True)
    clips = {}
    missing = {k: audio_dir / f"{k}.wav" for k in ALERT_PHRASES}
    missing = {k: p for k, p in missing.items() if not p.exists()}
    if missing:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            for kind, path in missing.items():
                engine.save_to_file(ALERT_PHRASES[kind], str(path))
            engine.runAndWait()
        except Exception:
            pass
    for kind in ALERT_PHRASES:
        path = audio_dir / f"{kind}.wav"
        if path.exists():
            clips[kind] = path
    return clips


class AudioPlayer:
    """Fire-and-forget WAV playback. One clip at a time; requests arriving
    while a clip is playing are dropped (the cooldown makes repeats cheap)."""

    def __init__(self, clips: dict[str, Path] | None = None, enabled: bool = True):
        self.enabled = enabled
        self.clips = clips if clips is not None else (synthesize_clips() if enabled else {})
        self._busy = threading.Lock()

    def play(self, kind: str, text: str | None = None):
        if not self.enabled:
            return
        if not self._busy.acquire(blocking=False):
            return  # something already playing
        threading.Thread(target=self._play_blocking, args=(kind, text),
                         daemon=True).start()

    def _play_blocking(self, kind: str, text: str | None = None):
        try:
            path = self.clips.get(kind)
            if platform.system() == "Windows":
                import winsound
                if path:
                    winsound.PlaySound(str(path), winsound.SND_FILENAME)
                else:
                    winsound.Beep(1200, 350)
                    winsound.Beep(900, 350)
                return
            # Linux (Spark): speak the scene-specific text directly —
            # espeak-ng starts in ~0.2s, so dynamic phrases stay on-the-spot
            if text:
                import shutil
                if shutil.which("espeak-ng"):
                    try:
                        subprocess.run(["espeak-ng", "-s", "165", "-a", "180", text],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=15)
                        return
                    except Exception:
                        pass
            if path:
                for player in (["aplay", "-q"], ["paplay"], ["ffplay", "-nodisp", "-autoexit",
                                                             "-loglevel", "quiet"]):
                    try:
                        subprocess.run(player + [str(path)], check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       timeout=15)
                        break
                    except Exception:
                        continue
        except Exception:
            pass
        finally:
            self._busy.release()


# --------------------------------------------------------------------------
# Alert state machine
# --------------------------------------------------------------------------

@dataclass
class _HazardState:
    consecutive: int = 0
    last_fired: float = 0.0
    total_fired: int = 0


class AlertEngine:
    """Feed it the set of active hazard types once per detection pass;
    it decides when an alarm actually fires."""

    def __init__(
        self,
        trigger_frames: int = config.ALERT_TRIGGER_FRAMES,
        cooldown_sec: float = config.ALERT_COOLDOWN_SEC,
        events_path: Path | None = config.EVENTS_LOG,
        player: AudioPlayer | None = None,
        clock=time.time,
    ):
        self.trigger_frames = max(trigger_frames, 1)
        self.cooldown_sec = cooldown_sec
        self.events_path = Path(events_path) if events_path else None
        self.player = player
        self.clock = clock
        self._states: dict[str, _HazardState] = {k: _HazardState() for k in ALERT_PHRASES}
        self._lock = threading.Lock()
        self.recent: list[dict] = []  # last fired alerts, newest last (dashboard)
        if self.events_path:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def update(self, active: set[str], detail: dict | None = None,
               messages: dict[str, str] | None = None) -> list[dict]:
        """One detection pass. `active` = hazard types present this pass.
        `messages` optionally overrides the spoken/logged phrase per type with
        a scene-specific one composed from detection geometry (still instant —
        no model call). Returns the alerts that fired now (usually empty)."""
        fired = []
        now = self.clock()
        with self._lock:
            for kind, state in self._states.items():
                if kind in active:
                    state.consecutive += 1
                    if (state.consecutive >= self.trigger_frames
                            and now - state.last_fired >= self.cooldown_sec):
                        state.last_fired = now
                        state.total_fired += 1
                        event = {
                            "time": now,
                            "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                            "type": kind,
                            "message": (messages or {}).get(kind) or ALERT_PHRASES[kind],
                            **(detail or {}),
                        }
                        fired.append(event)
                else:
                    state.consecutive = 0
            for event in fired:
                self.recent.append(event)
            del self.recent[:-50]
        for event in fired:
            self._log(event)
            if self.player:
                self.player.play(event["type"], text=event["message"])
        return fired

    def fire_now(self, kind: str, message: str | None = None,
                 detail: dict | None = None) -> dict | None:
        """Fire one alert immediately (used by verified escalations, e.g. the
        VLM-confirmed height hazard). Respects the per-type cooldown but not
        the frame debounce. Returns the event, or None if still cooling down."""
        now = self.clock()
        with self._lock:
            st = self._states[kind]
            if now - st.last_fired < self.cooldown_sec:
                return None
            st.last_fired = now
            st.total_fired += 1
            event = {
                "time": now,
                "iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "type": kind,
                "message": message or ALERT_PHRASES[kind],
                **(detail or {}),
            }
            self.recent.append(event)
            del self.recent[:-50]
        self._log(event)
        if self.player:
            self.player.play(kind, text=event["message"])
        return event

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {k: s.total_fired for k, s in self._states.items()}

    def reset(self):
        """New session (see SessionState.reset_session): fresh debounce
        state and a cleared dashboard log/counts. The events.jsonl file on
        disk is untouched — this only resets what's held in memory for the
        live display, not the permanent audit trail."""
        with self._lock:
            self._states = {k: _HazardState() for k in ALERT_PHRASES}
            self.recent = []

    def _log(self, event: dict):
        if not self.events_path:
            return
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass


if __name__ == "__main__":
    clips = synthesize_clips()
    print(f"Synthesized clips: { {k: str(v) for k, v in clips.items()} }")
    player = AudioPlayer(clips)
    for kind in ALERT_PHRASES:
        print(f"Playing: {kind}")
        player._play_blocking(kind)
