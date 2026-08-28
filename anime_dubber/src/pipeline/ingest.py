from __future__ import annotations
import logging
from pathlib import Path
from src.core.paths import JobPaths
log = logging.getLogger(__name__)

async def run_ingest(job: Path, cfg: dict, video_path: Path) -> None:
    paths = JobPaths(job)
    paths.create_all()
    from src.ingest.media import extract_audio
    from src.speech.vad import run_vad
    dst = paths.source / "scene.mp4"
    if not dst.exists():
        import shutil
        shutil.copy2(video_path, dst)
    extract_audio(dst, paths.audio / "original.wav")
    await run_vad(str(paths.audio / "original.wav"), str(paths.audio / "vad.json"))
    log.info("Ingest done")
