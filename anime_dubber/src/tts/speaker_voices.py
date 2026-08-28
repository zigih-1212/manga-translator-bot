"""Automatic speaker-to-voice mapping for edge-tts."""
from __future__ import annotations
import logging

log = logging.getLogger(__name__)

# Edge-tts voices by language and gender/style
VOICE_POOLS = {
    "ru": {
        "male": [
            "ru-RU-DmitryNeural",      # deep, serious
            "ru-RU-YuriyNeural",        # younger, energetic
        ],
        "female": [
            "ru-RU-SvetlanaNeural",     # mature, calm
            "ru-RU-DariyaNeural",       # younger, friendly
        ],
    },
    "en": {
        "male": [
            "en-US-GuyNeural",          # deep, serious
            "en-US-DavisNeural",        # younger
            "en-GB-RyanNeural",         # British accent
        ],
        "female": [
            "en-US-AriaNeural",         # natural, friendly
            "en-US-JennyNeural",        # younger
            "en-GB-SoniaNeural",        # British accent
        ],
    },
    "ja": {
        "male": [
            "ja-JP-KeitaNeural",
            "ja-JP-DaichiNeural",
        ],
        "female": [
            "ja-JP-NanamiNeural",
            "ja-JP-AoiNeural",
        ],
    },
    "ko": {
        "male": [
            "ko-KR-InJoonNeural",
            "ko-KR-JinhoNeural",
        ],
        "female": [
            "ko-KR-SunHiNeural",
            "ko-KR-SoYeonNeural",
        ],
    },
}


def assign_voices(speakers: list[str], lang: str = "ru") -> dict[str, str]:
    """Assign unique voices to each speaker automatically.

    Args:
        speakers: List of speaker IDs (e.g., ["SPEAKER_00", "SPEAKER_01"])
        lang: Target language code

    Returns:
        Dict mapping speaker_id -> voice_name
    """
    pool = VOICE_POOLS.get(lang, VOICE_POOLS["ru"])

    # Alternate male/female for variety
    mapping = {}
    male_idx = 0
    female_idx = 0

    for i, speaker in enumerate(speakers):
        if i % 2 == 0:
            # Even index -> male voice
            voices = pool["male"]
            voice = voices[male_idx % len(voices)]
            male_idx += 1
        else:
            # Odd index -> female voice
            voices = pool["female"]
            voice = voices[female_idx % len(voices)]
            female_idx += 1
        mapping[speaker] = voice
        log.info(f"Speaker {speaker} -> {voice}")

    return mapping


def get_voice_for_speaker(speaker: str, lang: str = "ru", gender: str = "male") -> str:
    """Get a specific voice for a speaker."""
    pool = VOICE_POOLS.get(lang, VOICE_POOLS["ru"])
    voices = pool.get(gender, pool["male"])
    # Simple hash-based selection for consistency
    idx = hash(speaker) % len(voices)
    return voices[idx]
