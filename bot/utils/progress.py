import asyncio
from typing import Optional
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest

from bot.utils import telegram_helpers


class ProgressReporter:
    def __init__(self, chat_id: int, manga_id: str, chapter: str):
        self.chat_id = chat_id
        self.key = (chat_id, manga_id, chapter)
        self._message_id: Optional[int] = None
        self._last_sent: float = 0.0
        self._min_interval = 1.5

    @property
    def _bot(self) -> Optional[Bot]:
        return telegram_helpers._bot

    async def start(self, text: str) -> None:
        bot = self._bot
        if bot is None:
            return
        try:
            msg = await bot.send_message(self.chat_id, text)
            self._message_id = msg.message_id
            self._last_sent = asyncio.get_event_loop().time()
        except Exception:
            pass

    async def update(self, text: str) -> None:
        bot = self._bot
        if bot is None or self._message_id is None:
            return
        now = asyncio.get_event_loop().time()
        if now - self._last_sent < self._min_interval:
            return
        try:
            await bot.edit_message_text(text, chat_id=self.chat_id, message_id=self._message_id)
            self._last_sent = now
        except TelegramBadRequest:
            pass
        except Exception:
            pass

    async def finish(self, text: str) -> None:
        bot = self._bot
        if bot is None:
            return
        if self._message_id is not None:
            try:
                await bot.edit_message_text(text, chat_id=self.chat_id, message_id=self._message_id)
            except Exception:
                pass
        else:
            try:
                await bot.send_message(self.chat_id, text)
            except Exception:
                pass


_reporters: dict[tuple, ProgressReporter] = {}


def get_reporter(chat_id: int, manga_id: str, chapter: str) -> ProgressReporter:
    key = (chat_id, manga_id, chapter)
    if key not in _reporters:
        _reporters[key] = ProgressReporter(chat_id, manga_id, chapter)
    return _reporters[key]


def clear_reporter(chat_id: int, manga_id: str, chapter: str) -> None:
    key = (chat_id, manga_id, chapter)
    _reporters.pop(key, None)