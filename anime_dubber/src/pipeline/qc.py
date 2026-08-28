"""QC pipeline stage."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List

from ..core.paths import JobPaths
from ..core.checkpoint import mark_stage, is_stage_done
from ..core.runner import run_stage
from .timing_qc import timing_info, timing_status
from .audio_qc import analyze_audio, audio_qc_checks, overall_status
from .text_qc import token_overlap, normalize_text
from .scorer import decide, score_from_checks

log = logging.getLogger(__name__)


async def run_qc(runner, job_dir: Path) -> None:
    """QC stage: timing, audio quality, text accuracy."""
    paths = JobPaths(job_dir)
    manifest = job_dir / "manifest.json"
    
    # Timing QC
    # For each chapter, check duration ratio
    # This is a stub - real implementation would check each chapter
    
    # Audio QC
    # Check for clipping, loudness, etc.
    
    # Text QC
    # Compare translated text with ASR of generated audio
    
    # Overall status
    # mark_stage(job_dir, "qc", "done")
    
    log.info("QC stage complete (stub)")
    pass