from __future__ import annotations

def duration_ratio(source_start: float, source_end: float, generated_duration: float) -> float:
    source_dur = max(source_end - source_start, 0.001)
    return generated_duration / source_dur


def timing_status(ratio: float) -> str:
    if 0.85 <= ratio <= 1.15:
        return "good"
    elif ratio <= 1.35 or ratio >= 0.65:
        return "acceptable"
    elif ratio <= 1.70 or ratio >= 0.59:
        return "suspicious"
    return "critical"


def timing_info(source_start: float, source_end: float, target_duration: float) -> dict:
    source_dur = max(source_end - source_start, 0.001)
    ratio = target_duration / source_dur
    return {
        "source_duration": max(source_end - source_start, 0.001),
        "target_duration": target_duration,
        "ratio": ratio,
        "status": "good" if 0.85 <= target_duration / max(source_end - source_start, 0.001) <= 1.15
        else "acceptable" if 0.65 <= target_duration / max(source_end - source_start, 0.001) <= 1.35
        else "suspicious" if 0.59 <= target_duration / max(source_end - source_start, 0.001) <= 1.70
        else "critical",
    }


def format_timing_info(info: dict) -> str:
    return f"src={info['source_start']:.2f}-{info['source_end']:.2f} ({info['source_duration']:.2f}s) tgt={info['target_duration']:.2f}s ratio={info['ratio']:.2f} [{info['status']}]"


def format_duration(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{seconds:06.3f}"


if __name__ == "__main__":
    info = {"source_start": 0.0, "source_end": 2.5, "target_duration": 2.8}
    info["ratio"] = 2.8 / 2.5
    info["status"] = "acceptable" if 1.15 < 2.8/2.5 <= 1.35 else "good"
    print(f"Ratio: {2.8/2.5:.2f}, Status: acceptable")