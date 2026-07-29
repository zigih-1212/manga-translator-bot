import json
from pathlib import Path
from collections import Counter
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


def save_translations(manga_id: str, title: str, chapter: str, pairs: list[dict]):
    data = _load()
    entry = data.setdefault(manga_id, {"title": title, "glossary": {}, "chapters": {}})
    entry["title"] = title
    entry["chapters"][chapter] = [{"ko": p.get("ko", ""), "ru": p.get("ru", "")} for p in pairs if p.get("ko") and p.get("ru")]
    _update_glossary(entry)
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
