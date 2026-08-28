"""Phase 2: Speech understanding pipeline."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from ..core.paths import JobPaths
from ..core.runner import run_stage
from ..core.checkpoint import mark_stage, is_stage_done
from .asr import run_transcribe
from .diarization import run_diarization
from .vad import run_vad
from .alignment import run_alignment
from .references import create_reference_bank

log = logging.getLogger(__name__)


async def run_speech_understanding(runner, job_dir: Path) -> None:
    """Phase 2: ASR, alignment, diarization, speaker mining."""
    paths = JobPaths(job_dir)
    manifest = job_dir / "manifest.json"
    
    # VAD
    await run_stage(runner.job_dir, "vad", lambda r: run_vad(paths.get_audio_path("original.wav"), paths.get_audio_path("vad.json")))
    
    # ASR
    await run_stage(runner.job_dir, "asr", lambda r: run_transcribe(
        paths.get_audio_path("original.wav"),
        paths.transcript / "asr.json",
    ))
    
    # Alignment
    await run_stage(runner.job_dir, "alignment", lambda r: run_alignment(
        paths.get_audio_path("original.wav"),
        paths.transcript / "asr.json",
        paths.transcript / "aligned.json",
    ))
    
    # Diarization
    await run_stage(runner.job_dir, "diarization", lambda r: run_diarization(
        paths.get_audio_path("original.wav"),
        paths.speakers / "diarization.json",
    ))
    
    # Speaker clustering + reference extraction
    await run_stage(runner.job_dir, "speaker_mining", lambda r: extract_references(
        job_dir,
        paths.speakers / "diarization.json",
        paths.speakers,
    ))
    
    log.info("Speech understanding phase complete")


async def run_speech_understanding(runner, job_dir: Path) -> None:
    paths = JobPaths(job_dir)
    
    # VAD
    await run_stage(job_dir, "vad", lambda r: run_vad(
        paths.get_audio_path("original.wav"),
        paths.get_audio_path("vad.json"),
    ))
    
    # ASR
    await run_stage(job_dir, "asr", lambda r: run_transcribe(
        paths.get_audio_path("original.wav"),
        paths.transcript / "asr.json",
    ))
    
    # Alignment
    await run_stage(job_dir, "alignment", lambda r: run_alignment(
        paths.get_audio_path("original.wav"),
        paths.transcript / "asr.json",
        paths.transcript / "aligned.json",
    ))
    
    # Diarization
    await run_stage(job_dir, "diarization", lambda r: run_diarization(
        paths.get_audio_path("original.wav"),
        paths.speakers / "diarization.json",
    ))
    
    # Speaker mining
    await run_stage(job_dir, "speaker_mining", lambda r: extract_references(
        job_dir,
        paths.speakers / "diarization.json",
        paths.speakers,
    ))
    
    log.info("Speech understanding phase complete")