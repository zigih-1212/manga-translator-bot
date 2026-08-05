import json
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv()  # CWD
load_dotenv(BASE_DIR / "cfg" / ".env")
load_dotenv(BASE_DIR / ".env")
load_dotenv(Path.cwd() / ".env")
load_dotenv(Path.cwd() / "cfg" / ".env")

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

def _deep_merge(target: dict, source: dict) -> dict:
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            target[k] = _deep_merge(target[k], v)
        else:
            target[k] = v
    return target

DEFAULT_CONFIG = {
    "webfandom": {"base_url": "https://webfandom.ru", "team_name": ""},
    "mangadex": {"base_url": "https://api.mangadex.org", "image_url": "https://uploads.mangadex.org"},
    "llm": {"provider": "remote", "model": "google/gemini-2.0-flash-lite", "fallback_provider": "deep-translator", "max_context_tokens": 32000, "temperature": 0.3, "vision": "off", "vision_model": ""},
    "fonts": {"dialogue": "fonts/anime/Anime Font.ttf", "sfx": "fonts/bring_me_a_helicopter/Bring Me A Helicopter!.otf", "narration": "fonts/wister_lilya/Wister Lilya.otf"},
    "translation": {"sfx_mode": "english_reference", "auto_publish": False, "max_pages_per_batch": 20, "post_replace": {}},
    "ocr": {"paddle": False, "colab_priority": False},
    "inpaint": {"nano_banana": False, "nano_banana_model": "gemini-3.1-flash-image"},
    "titles": [],
    "telegram": {},
    "chapters": {},
    "webhooks": {"urls": []},
}
CONFIG = DEFAULT_CONFIG.copy()
loaded_config = _load_json(CONFIG_PATH, {})
_deep_merge(CONFIG, loaded_config)
GLOSSARY = _load_json(GLOSSARY_PATH, {})
FONTS = _load_json(FONTS_PATH, {})

TG_BOT_TOKEN = os.getenv("TG_BOT_TOKEN") or CONFIG.get("telegram", {}).get("bot_token") or CONFIG.get("telegram", {}).get("token") or ""
TG_API_ID = int(os.getenv("TG_API_ID") or CONFIG.get("telegram", {}).get("api_id") or "0")
TG_API_HASH = os.getenv("TG_API_HASH") or CONFIG.get("telegram", {}).get("api_hash") or ""
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or CONFIG.get("llm", {}).get("groq_api_key") or ""
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or CONFIG.get("llm", {}).get("gemini_api_key") or ""
WEBFANDOM_ACCESS_TOKEN = os.getenv("WEBFANDOM_ACCESS_TOKEN", "")
WEBFANDOM_REFRESH_TOKEN = os.getenv("WEBFANDOM_REFRESH_TOKEN", "")
REMOTE_SERVER_URL = os.getenv("REMOTE_SERVER_URL") or os.getenv("COLAB_URL") or CONFIG.get("llm", {}).get("remote_server_url") or ""
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or CONFIG.get("llm", {}).get("openrouter_api_key") or ""
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY") or CONFIG.get("llm", {}).get("deepl_api_key") or ""
TG_PROXY_URL = os.getenv("TG_PROXY_URL") or CONFIG.get("telegram", {}).get("proxy_url") or ""

TEMP_DIR = BASE_DIR / "temp"
TEMP_DIR.mkdir(exist_ok=True)


def save_config():
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(CONFIG, ensure_ascii=False, indent=2)
    tmp_path = CONFIG_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, CONFIG_PATH)
    except Exception:
        # Fallback: если атомарная запись не удалась, пишем напрямую
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(data)


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


def load_glossary() -> dict:
    """Загрузить глоссарий из файла (characters + terms)."""
    return _load_json(GLOSSARY_PATH, {"characters": {}, "terms": {}})


def save_glossary(glossary: dict):
    """Сохранить глоссарий в файл атомарно."""
    GLOSSARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(glossary, ensure_ascii=False, indent=2)
    tmp_path = GLOSSARY_PATH.with_suffix(".json.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, GLOSSARY_PATH)
    except Exception:
        with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
            f.write(data)
    # Update module-level reference
    global GLOSSARY
    GLOSSARY = glossary
