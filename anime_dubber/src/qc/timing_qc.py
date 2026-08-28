from __future__ import annotations
def duration_ratio(s,e,d): return d/max(e-s,0.001)
def timing_status(r: float) -> str:
    if 0.85 <= r <= 1.15: return "good"
    if 0.65 <= r <= 1.70: return "suspicious" if r>1.35 else "acceptable"
    return "critical"
def timing_info(s,e,d): r=duration_ratio(s,e,d); return {"source_duration":max(e-s,0.001),"target_duration":d,"ratio":r,"status":timing_status(r)}
