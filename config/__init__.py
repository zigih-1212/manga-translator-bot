import json
from pathlib import Path
from dotenv import load_dotenv
import os

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / "config" / ".env")
load_dotenv(BASE_DIR / ".env")

CONFIG_PATH = BASE_DIR / "config" / "config.json"
GLOSSARY_PATH = BASE_DIR / "config" / "glossary.json"
FONTS_PATH = BASE_DIR / "config" / "fonts.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
    GLOSSARY = json.load(f)

with open(FONTS_PATH, "r", encoding="utf-8") as f:
    FONTS = json.load(f)

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
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, indent=2)
