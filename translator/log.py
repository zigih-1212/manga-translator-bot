import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

DATA_DIR = os.environ.get("DATA_DIR", "data")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            entry["exc"] = self.formatException(record.exc_info)
        for key in ("page", "chapter", "manga", "provider", "model", "duration_ms", "status"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry, ensure_ascii=False)


def setup_logging(level=logging.INFO):
    logger = logging.getLogger("manga_translator")
    logger.setLevel(level)
    if logger.handlers:
        return logger

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(
        logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(console)

    try:
        log_dir = Path(DATA_DIR) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        json_handler = RotatingFileHandler(
            str(log_dir / "app.jsonl"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        json_handler.setLevel(level)
        json_handler.setFormatter(JsonFormatter())
        logger.addHandler(json_handler)
    except Exception as e:
        sys.stderr.write(f"JSON log init failed: {e}\n")

    return logger


log = setup_logging()
