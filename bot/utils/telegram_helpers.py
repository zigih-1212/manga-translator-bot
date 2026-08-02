from aiogram import Bot
from aiogram.types import FSInputFile
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

_bot: Bot | None = None


def set_bot(bot: Bot) -> None:
    global _bot
    _bot = bot


async def send_text(chat_id: int, text: str) -> bool:
    """Safely send text message to chat. Returns True if successful."""
    if _bot is None:
        logger.warning("Bot not initialized, cannot send message")
        return False
    try:
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
        await _bot.send_document(chat_id, document=FSInputFile(str(path)), caption=caption)
        return True
    except Exception as e:
        logger.exception(f"Failed to send document to {chat_id}: {e}")
        return False