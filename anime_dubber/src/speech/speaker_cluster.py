from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Dict
log = logging.getLogger(__name__)

def cluster_speakers(diarization: dict, embeddings_dir: str | Path | None = None, threshold: float = 0.8) -> Dict[str, str]:
    """Stub: identity mapping. Real impl would cluster embeddings with AgglomerativeClustering."""
    speakers = {s["speaker"] for s in diarization.get("segments", [])}
    return {s: s for s in speakers}

def merge_speakers(diarization_json: str | Path, mappings: Dict[str, str], output_json: str | Path) -> None:
    with open(diarization_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    for seg in data.get("segments", []):
        if seg["speaker"] in mappings:
            seg["speaker"] = mappings[seg["speaker"]]
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
