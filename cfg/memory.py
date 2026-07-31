import json
from pathlib import Path
from collections import Counter
import re
from . import DATA_DIR

MEMORY_PATH = DATA_DIR / "memory.json"


def _load():
    if MEMORY_PATH.exists():
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data: dict):
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _extract_speaker(text: str, glossary: dict) -> str | None:
    """Extract speaker name from text like 'Name: text' or from glossary."""
    # Pattern: "Name:" or "Name：" at start
    m = re.match(r'^([\w\s\-\.]{1,20})[:：]', text.strip())
    if m:
        name = m.group(1).strip()
        # Check if name is in glossary characters
        for char_name in glossary.get("characters", {}):
            if name in char_name or char_name in name:
                return char_name
        return name
    return None


def _update_character_profile(entry: dict, speaker: str, text: str, translation: str):
    """Update character voice profile from translation pairs."""
    if not speaker or speaker == "unknown":
        return
    chars = entry.setdefault("characters", {})
    prof = chars.setdefault(speaker, {
        "tone": "",
        "examples_ko": [],
        "examples_ru": [],
        "speech_patterns": [],
        "politeness": "neutral",
    })
    # Store examples (max 5 each)
    if len(prof["examples_ko"]) < 5:
        prof["examples_ko"].append(text.strip())
    if len(prof["examples_ru"]) < 5:
        prof["examples_ru"].append(translation.strip())
    # Detect politeness
    if any(p in text for p in ["입니다", "하십시오", "시요", "masu", "desu"]):
        prof["politeness"] = "polite"
    elif any(p in text for p in ["야", "다", "이야", "da", "yo"]):
        prof["politeness"] = "casual"
    # Auto-generate tone summary
    if prof["examples_ko"]:
        prof["tone"] = f"{prof['politeness']}, {len(prof['examples_ko'])} samples"


def save_translations(manga_id: str, title: str, chapter: str, pairs: list[dict]):
    data = _load()
    entry = data.setdefault(manga_id, {"title": title, "glossary": {}, "characters": {}, "chapters": {}})
    entry["title"] = title
    entry["chapters"][chapter] = [{"ko": p.get("ko", ""), "ru": p.get("ru", "")} for p in pairs if p.get("ko") and p.get("ru")]
    _update_glossary(entry)
    # Update character profiles
    glossary = entry.get("glossary", {})
    for p in pairs:
        ko = p.get("ko", "").strip()
        ru = p.get("ru", "").strip()
        if ko and ru:
            speaker = p.get("speaker") or _extract_speaker(ko, glossary)
            if speaker:
                _update_character_profile(entry, speaker, ko, ru)
    _save(data)


def _update_glossary(entry: dict):
    counter = Counter()
    for ch_data in entry["chapters"].values():
        for p in ch_data:
            ko = p.get("ko", "").strip()
            ru = p.get("ru", "").strip()
            if ko and ru and len(ko) > 1:
                counter[(ko, ru)] += 1
    entry["glossary"] = {ko: ru for (ko, ru), count in counter.most_common(50) if count >= 2}


def get_context(manga_id: str, max_chapters: int = 2) -> str:
    data = _load()
    entry = data.get(manga_id)
    if not entry:
        return ""
    chapters = entry.get("chapters", {})
    sorted_chs = sorted(chapters.keys(), key=lambda x: float(x))
    recent = sorted_chs[-max_chapters:]
    parts = []
    for ch in recent:
        pairs = chapters[ch]
        samples = [f"  {p['ko']} → {p['ru']}" for p in pairs[:10]]
        parts.append(f"Chapter {ch}:\n" + "\n".join(samples))
    glossary = entry.get("glossary", {})
    if glossary:
        terms = "\n".join(f"  {ko} → {ru}" for ko, ru in list(glossary.items())[:20])
        parts.insert(0, f"Known terms from previous chapters:\n{terms}")
    return "\n\n".join(parts)


def get_glossary(manga_id: str) -> dict:
    data = _load()
    return data.get(manga_id, {}).get("glossary", {})


def get_character_profiles(manga_id: str) -> dict:
    """Return character voice profiles for prompt injection."""
    data = _load()
    entry = data.get(manga_id, {})
    chars = entry.get("characters", {})
    # Return only useful profiles
    return {
        name: {
            "tone": p.get("tone", ""),
            "examples_ru": p.get("examples_ru", [])[:3],
            "politeness": p.get("politeness", "neutral"),
        }
        for name, p in chars.items()
        if p.get("examples_ru")
    }
