import json
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "cfg" / ".env")
load_dotenv(BASE_DIR / ".env")

DATA_DIR = Path(os.getenv("DATA_DIR", "") or str(BASE_DIR / "cfg"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = DATA_DIR / "config.json"
GLOSSARY_PATH = DATA_DIR / "glossary.json"
FONTS_PATH = DATA_DIR / "fonts.json"


def _load_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


DEFAULT_CONFIG = {
    "webfandom": {"base_url": "https://webfandom.ru", "team_name": ""},
    "mangadex": {"base_url": "https://api.mangadex.org", "image_url": "https://uploads.mangadex.org"},
    "llm": {"provider": "colab", "model": "google/gemini-2.0-flash-lite", "fallback_provider": "deep-translator", "max_context_tokens": 32000, "temperature": 0.3},
    "fonts": {"dialogue": "fonts/anime/Anime Font.ttf", "sfx": "fonts/bring_me_a_helicopter/Bring Me A Helicopter!.otf", "narration": "fonts/wister_lilya/Wister Lilya.otf"},
    "translation": {"sfx_mode": "english_reference", "auto_publish": False, "max_pages_per_batch": 20, "post_replace": {}},
    "titles": [],
    "telegram": {},
    "chapters": {},
}
CONFIG = _load_json(CONFIG_PATH, DEFAULT_CONFIG)
GLOSSARY = _load_json(GLOSSARY_PATH, {})
FONTS = _load_json(FONTS_PATH, {})

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
TG_API_ID = int(os.getenv("TG_API_ID", "0"))
TG_API_HASH = os.getenv("TG_API_HASH")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
WEBFANDOM_ACCESS_TOKEN = os.getenv("WEBFANDOM_ACCESS_TOKEN")
WEBFANDOM_REFRESH_TOKEN = os.getenv("WEBFANDOM_REFRESH_TOKEN")
COLAB_URL = os.getenv("COLAB_URL", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TG_PROXY_URL = os.getenv("TG_PROXY_URL", "")

TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)


def save_config():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)


_REQUIRED_CONFIG_KEYS = {
    "mangadex": ["base_url"],
    "llm": ["provider", "model"],
}


def validate_config():
    import logging
    log = logging.getLogger("manga_translator")
    issues = []
    if not TG_BOT_TOKEN:
        issues.append("TG_BOT_TOKEN не установлен — бот не запустится")
    if not GEMINI_API_KEY and not GROQ_API_KEY and not OPENROUTER_API_KEY:
        issues.append("Нет ни одного API ключа (GEMINI/GROQ/OPENROUTER) — перевод не будет работать")
    for section, keys in _REQUIRED_CONFIG_KEYS.items():
        cfg_section = CONFIG.get(section, {})
        for key in keys:
            if not cfg_section.get(key):
                issues.append(f"CONFIG.{section}.{key} отсутствует")
    for issue in issues:
        log.warning("Config issue: %s", issue)
    if not issues:
        log.info("Config validation OK")
    return len(issues) == 0
