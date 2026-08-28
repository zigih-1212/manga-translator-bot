from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List

log = logging.getLogger(__name__)


class CharacterResolver:
    """
    Resolves speaker clusters to character identities and assigns voices.
    """

    def __init__(self, characters_dir: str | Path = "jobs/characters"):
        self.characters_dir = Path(characters_dir)
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.characters_file = Path(characters_dir) / "characters.json"
        self.mappings_file = Path(characters_dir) / "mappings.json"

        self.characters = self._load_characters()
        self.mappings = self._load_mappings()

    def _load_characters(self) -> dict:
        if self.characters_file.exists():
            with open(self.characters_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_characters(self):
        with open(self.characters_file, "w", encoding="utf-8") as f:
            json.dump(self.characters, f, ensure_ascii=False, indent=2)

    def _load_mappings(self) -> dict:
        if self.mappings_file.exists():
            with open(self.mappings_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_mappings(self):
        with open(self.mappings_file, "w", encoding="utf-8") as f:
            json.dump(self.mappings, f, ensure_ascii=False, indent=2)

    def resolve_speaker(self, speaker_id: str, source_name: str) -> dict:
        """
        Resolve a speaker cluster to a character identity and voice.
        """
        key = f"{speaker_id}"
        
        if key in self.mappings:
            return self.mappings[key]

        # Auto-assign new character
        char_id = f"char_{len(self.characters) + 1:03d}"
        voice_id = f"voice_{len(self.characters) + 1:03d}"
        
        char_data = {
            "character_id": char_id,
            "voice_id": voice_id,
            "source_speaker": speaker_id,
            "voice_name": f"Character {len(self.characters) + 1}",
            "source_lang": "en",
        }
        
        self.mappings[key] = char_id
        self.characters[char_id] = {
            "character_id": char_id,
            "voice_id": voice_id,
            "source_speaker": speaker_id,
            "voice_name": f"Character {len(self.characters) + 1}",
            "source_lang": "en",
        }
        self._save_mappings()
        self._save_characters()
        
        return {
            "character_id": char_id,
            "voice_id": voice_id,
            "name": f"Character {len(self.characters)}",
            "source_lang": "en",
        }
    
    def get_character(self, character_id: str) -> Optional[dict]:
        return self.characters.get(character_id)
    
    def set_voice(self, character_id: str, voice_id: str):
        if character_id in self.characters:
            self.characters[character_id]["voice_id"] = voice_id
            self._save_characters()
    
    def list_characters(self) -> List:
        return list(self.characters.values())


def load_character_mappings(mappings_file: Path) -> Dict:
    if mappings_file.exists():
        with open(mappings_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_character_mappings(mappings_file: Path, mappings: dict):
    mappings_file.parent.mkdir(parents=True, exist_ok=True)
    with open(mappings_file, "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)


def load_characters(characters_file: Path) -> dict:
    if characters_file.exists():
        with open(characters_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_characters(characters_file: Path, characters: dict):
    characters_file.parent.mkdir(parents=True, exist_ok=True)
    with open(characters_file, "w", encoding="utf-8") as f:
        json.dump(characters, f, ensure_ascii=False, indent=2)


def get_character_voice(character_id: str, mappings: dict) -> str:
    """Get voice ID for a character."""
    return mappings.get(character_id, "voice_01")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--diarization", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # Example usage
    resolver = CharacterResolver()
    mapping = {"SPEAKER_00": "char_001", "SPEAKER_01": "char_002"}
    with open(args.diarization, "r") as f:
        data = json.load(open(args.diarization, encoding="utf-8"))
    # Resolve
    print("Character resolver ready")