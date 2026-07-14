"""
Phase 2: LLM-generated safety report from the Phase 1 detection report dict.

Uses the Ollama server on the NVIDIA DGX Spark (config.OLLAMA_HOST, port
11434) through llm_client. The model preference chain (config.REPORT_MODELS,
default gpt-oss:120b -> qwen3:32b) resolves to whatever is actually pulled on
the server. A small built-in safety-regulation reference block is injected
into the prompt (lightweight RAG). Falls back to a templated markdown report
if the server or every model is unavailable, so the app never crashes.
"""

# Static safety reference snippets injected into the prompt so the report
# grounds its recommendations in named rules instead of generic advice.
SAFETY_REFERENCES = """\
- OSHA 1926.601 / ISO 5006: earthmoving machinery must maintain visibility of \
workers; reversing vehicles require a spotter or functioning alarm.
- Exclusion zone practice: a minimum 5 m clearance between workers on foot and \
operating heavy machinery; barriers or marked zones where practical.
- PDPA data minimization: monitoring records must avoid storing identifiable \
worker imagery; retain only aggregate counts and event timestamps."""


def _severity(report: dict) -> str:
    events = report.get("total_hazard_events", 0)
    frames = max(report.get("frames_processed", 1), 1)
    ratio = events / frames
    if events == 0:
        return "LOW"
    if ratio < 0.05:
        return "MODERATE"
    return "HIGH"


def _group_events(events: list, max_gap_sec: float = 1.5) -> list:
    """Merge hazard events close in time into continuous exposure intervals."""
    if not events:
        return []
    groups = []
    current = {"start": events[0]["timestamp_sec"], "end": events[0]["timestamp_sec"],
               "max_pairs": events[0]["pairs"]}
    for e in events[1:]:
        if e["timestamp_sec"] - current["end"] <= max_gap_sec:
            current["end"] = e["timestamp_sec"]
            current["max_pairs"] = max(current["max_pairs"], e["pairs"])
        else:
            groups.append(current)
            current = {"start": e["timestamp_sec"], "end": e["timestamp_sec"], "max_pairs": e["pairs"]}
    groups.append(current)
    return groups


def _format_events(report: dict, limit: int = 10) -> str:
    groups = _group_events(report.get("proximity_hazard_events", []))
    if not groups:
        return "None detected."
    lines = []
    for g in groups[:limit]:
        duration = g["end"] - g["start"]
        who = f"{g['max_pairs']} worker(s)" if g["max_pairs"] > 1 else "a worker"
        if duration < 0.5:
            lines.append(f"- At {g['start']:.1f}s: {who} within unsafe distance of operating heavy machinery")
        else:
            lines.append(f"- From {g['start']:.1f}s to {g['end']:.1f}s ({duration:.1f}s continuous): "
                         f"{who} within unsafe distance of operating heavy machinery")
    if len(groups) > limit:
        lines.append(f"- ... and {len(groups) - limit} further intervals")
    return "\n".join(lines)


def _format_incident_notes(report: dict) -> str:
    """Scene descriptions produced by the VLM at the moment each alert fired
    (live monitoring only). Injected as eyewitness-style notes so the report
    can describe the actual machinery and spatial situation, not just times."""
    notes = report.get("incident_notes") or []
    if not notes:
        return ""
    lines = [f"- [{n.get('iso', '')}] {n.get('type', 'hazard')}: {n.get('note', '')}"
             for n in notes if n.get("note")]
    if not lines:
        return ""
    return ("\nEYEWITNESS SCENE NOTES from the flagged moments "
            "(use these details in sections 1 and 2):\n" + "\n".join(lines) + "\n")


def _build_prompt(report: dict) -> str:
    return f"""You are a certified construction site safety officer writing a formal site safety report about footage reviewed from a construction site. Write about the SITE and the WORKERS — what happened, where the danger was, when. Do NOT mention AI, models, detection systems, algorithms, video analysis software, or how the footage was processed. Be direct and specific. No filler, no hedging.

SITE OBSERVATIONS (authoritative — use these facts):
- Footage: {report.get('video', 'unknown')}, {report.get('frames_processed', 0)} frames reviewed
- Workers observed on site throughout the footage
- Heavy machinery/vehicles operating on site throughout the footage
- Assessed severity: {_severity(report)}

UNSAFE PROXIMITY LOG (worker too close to operating heavy machinery):
{_format_events(report)}
{_format_incident_notes(report)}
SAFETY REFERENCES (cite these where relevant):
{SAFETY_REFERENCES}

Produce EXACTLY this Markdown structure, filling in content:

## Site Safety Report

**Severity: {_severity(report)}**

### 1. Summary
(2-3 direct sentences about site conditions: workers operating alongside heavy machinery, how many unsafe proximity intervals occurred and their overall pattern.)

### 2. Hazards Identified
(Bullet list from the proximity log: each bullet states when, for how long, and that a worker was inside the unsafe zone of operating machinery. If none, one line stating no unsafe proximity was observed.)

### 3. Recommended Actions
(Numbered list, most urgent first. Concrete site actions — exclusion zones, spotters, alarms, briefings — citing the safety references.)

### 4. Data Protection Note
(One short paragraph: this report stores no worker identity or imagery, only aggregate counts and event timestamps, per PDPA data minimization.)

Output only the report. Nothing before or after it."""


def _fallback_report(report: dict) -> str:
    severity = _severity(report)
    events = report.get("total_hazard_events", 0)
    n_intervals = len(_group_events(report.get("proximity_hazard_events", [])))
    hazard_section = (
        _format_events(report)
        if events else "No unsafe worker-machinery proximity was observed in this footage."
    )
    summary = (
        f"Workers were observed operating alongside heavy machinery in the reviewed footage "
        f"(\"{report.get('video', 'unknown')}\"). "
        + (f"{n_intervals} unsafe proximity interval(s) occurred in which workers were inside "
           f"the danger zone of operating machinery."
           if events else "No unsafe proximity between workers and machinery was observed.")
    )
    return f"""## Site Safety Report

**Severity: {severity}**

### 1. Summary
{summary}

### 2. Hazards Identified
{hazard_section}

### 3. Recommended Actions
1. Establish and enforce a marked exclusion zone of at least 5 m between workers on foot and operating machinery (exclusion zone practice).
2. Assign a dedicated spotter for reversing and slewing machinery and verify reversing alarms function (OSHA 1926.601 / ISO 5006).
3. Review the flagged intervals with the site supervisor and the workers involved.
4. Reinforce machinery proximity rules at the next toolbox talk.

### 4. Data Protection Note
This report contains no worker identity or imagery. Only aggregate counts and event timestamps are recorded, in line with PDPA data minimization principles.
"""


def hazard_intervals(report: dict) -> tuple[int, float]:
    """Return (number of grouped hazard intervals, total exposure seconds)."""
    groups = _group_events(report.get("proximity_hazard_events", []))
    exposure = sum(max(g["end"] - g["start"], 0.2) for g in groups)
    return len(groups), round(exposure, 1)


_ASCII_SWAPS = {"—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "·": "-", "…": "..."}


def report_to_pdf(markdown_text: str) -> bytes:
    """Render the markdown safety report as a simple formatted PDF."""
    from fpdf import FPDF

    text = markdown_text
    for k, v in _ASCII_SWAPS.items():
        text = text.replace(k, v)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            pdf.ln(3)
            continue
        if line.startswith("## "):
            pdf.set_font("helvetica", "B", 16)
            pdf.multi_cell(0, 9, line[3:], new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif line.startswith("### "):
            pdf.set_font("helvetica", "B", 12)
            pdf.multi_cell(0, 7, line[4:], new_x="LMARGIN", new_y="NEXT")
        else:
            bold = line.startswith("**") and line.rstrip().endswith("**")
            content = line.strip("*") if bold else line.replace("**", "")
            pdf.set_font("helvetica", "B" if bold else "", 10)
            pdf.multi_cell(0, 5.5, content, new_x="LMARGIN", new_y="NEXT")

    return bytes(pdf.output())


def _clean_llm_text(text: str) -> str:
    """Strip code fences and any <think>...</think> reasoning block that
    thinking models (qwen3, deepseek-r1) prepend."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("markdown").strip()
    return text


def generate_hazard_report(report: dict) -> str:
    try:
        from llm_client import get_client
        text = get_client().generate_report(_build_prompt(report))
        if text:
            text = _clean_llm_text(text)
            if "## Site Safety Report" in text:
                return text
    except Exception:
        pass
    return _fallback_report(report)


if __name__ == "__main__":
    sample = {
        "video": "sample_site_footage.mp4",
        "approach": "Plan A — Fine-tuned YOLOv8",
        "frames_processed": 300,
        "duration_sec": 10.0,
        "avg_fps": 30.0,
        "detections": {"worker": 3, "dangerous_vehicle": 2},
        "proximity_hazard_events": [
            {"frame": 120, "timestamp_sec": 4.0, "pairs": 1},
        ],
        "total_hazard_events": 1,
    }
    print(generate_hazard_report(sample))
