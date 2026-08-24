from aiogram import Bot
from aiogram.types import FSInputFile
from pathlib import Path
import asyncio
import logging

logger = logging.getLogger(__name__)

_bot: Bot | None = None

# Telegram limit: ~30 msg/sec globally, 20 msg/min per group.
# Simple token bucket: max 18 msgs / rolling 60s window.
_SEND_TIMES: list[float] = []
_RATE_LOCK = asyncio.Lock()


async def _rate_limit():
    """Sleep just enough to stay under Telegram's rate limits."""
    import time as _t
    async with _RATE_LOCK:
        now = _t.monotonic()
        while _SEND_TIMES and now - _SEND_TIMES[0] > 60:
            _SEND_TIMES.pop(0)
        if len(_SEND_TIMES) >= 18:
            wait = 60 - (now - _SEND_TIMES[0]) + 0.5
            logger.debug("Telegram rate-limit: sleeping %.1fs", wait)
            await asyncio.sleep(max(wait, 0.1))
        _SEND_TIMES.append(_t.monotonic())


def set_bot(bot: Bot) -> None:
    global _bot
    _bot = bot


async def send_text(chat_id: int, text: str) -> bool:
    """Safely send text message to chat. Returns True if successful."""
    if _bot is None:
        logger.warning("Bot not initialized, cannot send message")
        return False
    try:
        await _rate_limit()
        await _bot.send_message(chat_id, text)
        return True
    except Exception as e:
        logger.exception(f"Failed to send message to {chat_id}: {e}")
        return False


async def send_document(chat_id: int, path: Path, caption: str = "") -> bool:
    """Safely send document to chat. Returns True if successful."""
    if _bot is None:
        logger.warning("Bot not initialized, cannot send document")
        return False
    try:
        await _rate_limit()
        await _bot.send_document(chat_id, document=FSInputFile(str(path)), caption=caption)
        return True
    except Exception as e:
        logger.exception(f"Failed to send document to {chat_id}: {e}")
        return False