"""Phase 4: Translation + Adaptation + TTS synthesis."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict

from ..core.paths import JobPaths
from ..core.checkpoint import mark_stage, is_stage_done
from ..core.runner import run_stage
from ..translation.translator import translate_ja_ru, adapt_translation
from ..translation.timing import timing_info, timing_status
from ..tts.synthesis import TTSOrchestrator
from ..tts.cosyvoice3 import CosyVoice3Backend
from ..utils.progress import get_reporter, clear_reporter
from ..utils.telegram_helpers import send_text

log = logging.getLogger(__name__)


async def run_translation(runner, job_dir: Path) -> None:
    """Phase 4: Translation, adaptation, TTS synthesis."""
    paths = JobPaths(job_dir)
    
    # Load characters and mappings
    characters_file = job_dir / "characters.json"
    mappings_file = job_dir / "mappings.json"
    
    if not characters_file.exists():
        log.warning("No characters.json found, skipping translation")
        return
    
    with open(job_dir / "characters.json", "r", encoding="utf-8") as f:
        characters = json.load(f)
    
    with open(mappings_file, "r", encoding="utf-8") as f:
        mappings = json.load(f)
    
    # Load aligned transcript
    aligned_path = Path("jobs/episode/transcript/aligned.json")
    if not aligned_path.exists():
        log.warning("No aligned transcript found")
        return
    
    with open(aligned_path, "r", encoding="utf-8") as f:
        aligned = json.load(f)
    
    # Initialize TTS orchestrator
    tts = TTSOrchestrator()
    await tts.initialize()
    
    # Process each segment
    segments = aligned.get("segments", [])
    for segment in segments:
        speaker = segment.get("speaker", "SPEAKER_00")
        ja_text = segment.get("ja_text", "")
        if not ja_text:
            continue
        
        # Get character info
        char_id = f"SPEAKER_{speaker}"
        voice_id = mappings.get(speaker, "voice_01")
        
        # Translate
        ru_text = translate_ja_ru(segment.get("ja_text", ""))
        
        # Adapt for timing
        source_dur = segment.get("end", 0) - segment.get("start", 0)
        adapted = adapt_translation(source_start=segment.get("start", 0), 
                                     source_end=segment.get("end", 0),
                                     target_duration=segment.get("duration", 0))
        
        # TODO: TTS synthesis
        # This is where we'd call the TTS orchestrator
        
    log.info("Translation phase complete (stub)")


async def run_translation(runner, job_dir: Path) -> None:
    log.info("Translation phase - stub implementation")
    pass


async def run_translation(runner, job_dir: Path) -> None:
    pass