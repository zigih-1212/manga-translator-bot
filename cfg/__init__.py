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


CONFIG = _load_json(CONFIG_PATH, {"chapters": {}, "chat_id": None})
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
