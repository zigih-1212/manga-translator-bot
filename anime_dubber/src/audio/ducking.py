from __future__ import annotations
from pathlib import Path
import shutil, subprocess, logging
log=logging.getLogger(__name__)
def apply_ducking(original_audio: str|Path, speech_segments, output_path: str|Path, attenuation_db: float = 9.0, **kw) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # stub: copy original as ducked (no actual suppression)
    try: shutil.copy2(original_audio, output_path)
    except Exception: Path(output_path).write_bytes(b"")
