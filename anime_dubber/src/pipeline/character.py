"""Character system: speaker clustering, reference extraction, and voice assignment."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional

from ..core.paths import JobPaths
from ..core.checkpoint import mark_stage, is_stage_done
from ..core.runner import run_stage
from ..speech.references import create_reference_bank, extract_speaker_embeddings
from ..speech.speaker_cluster import cluster_speakers

log = logging.getLogger(__name__)


async def run_character_system(runner, job_dir: Path) -> None:
    """Phase 3: Speaker clustering, reference extraction, character resolution."""
    paths = JobPaths(job_dir)
    
    # Load diarization results
    diarization_path = paths.speakers / "diarization.json"
    if not diarization_path.exists():
        log.warning("No diarization.json found, skipping character system")
        return
    
    with open(diarization_path, "r", encoding="utf-8") as f:
        diarization = json.load(diarization_path)
    
    segments = diarization.get("segments", [])
    num_speakers = len(set(s["speaker"] for s in diarization.get("segments", [])))
    
    # Cluster speakers (if multiple similar speakers exist)
    log.info(f"Clustering {num_speakers} speakers...")
    clusters = cluster_speakers(diarization, "jobs/episode/speakers/embeddings")
    
    # Extract reference audio for each speaker
    log.info("Extracting reference audio for each speaker...")
    ref_bank = create_reference_bank(
        diarization.get("segments", []),
        "jobs/episode/audio/original.wav",
        "jobs/episode/speakers/references",
        per_speaker=3,
    )
    
    # Character resolver - map speaker clusters to characters
    from .character_resolver import CharacterResolver
    resolver = CharacterResolver()
    
    characters = {}
    for speaker_id in ref_bank:
        char_info = resolver.resolve_speaker(f"SPEAKER_{speaker_id}", "mangakakalot")
        characters[speaker_id] = char_info
    
    # Save characters.json and mappings.json
    characters_file = Path("jobs/episode/characters.json")
    mappings_file = Path("jobs/episode/mappings.json")
    
    with open("jobs/episode/characters.json", "w", encoding="utf-8") as f:
        json.dump(characters, f, ensure_ascii=False, indent=2)
    
    # Save mappings
    mappings = {k: v["character_id"] for k, v in characters.items()}
    with open("jobs/episode/mappings.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    log.info(f"Character system: {len(characters)} characters resolved")