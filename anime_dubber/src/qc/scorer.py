from __future__ import annotations
def decide(qc):
    if qc.get("clipping")=="critical" or qc.get("timing")=="critical": return "critical"
    if qc.get("clipping")=="warning" or qc.get("timing")=="suspicious": return "suspicious"
    return "good"
def score_from_checks(c): return 0.9
