from __future__ import annotations
import json, shutil
from pathlib import Path
async def align_transcript(audio_path: str | Path, transcript_json: str | Path, output_json: str | Path, language: str = "ja") -> dict:
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(transcript_json, output_json)
    with open(output_json, encoding="utf-8") as f: return json.load(f)
