from aiogram import Router
from aiogram.types import Message
from aiogram.filters import CommandStart, Command
from config import CONFIG, save_config

router = Router()


@router.message(CommandStart())
async def cmd_start(message: Message):
    CONFIG.setdefault("telegram", {})["chat_id"] = message.chat.id
    save_config()
    await message.answer(
        "Manga Translator Bot\n\n"
        "/add_title — добавить тайтл (поиск на MangaDex)\n"
        "/list — список тайтлов\n"
        "/translate — перевод главы\n"
        "/status — активные задачи\n"
        "/help — помощь\n\n"
        "Бот автоматически проверяет новые главы каждые 6ч."
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "Как пользоваться:\n\n"
        "1. /add_title — найди тайтл на MangaDex\n"
        "   Бот покажет доступные языки и главы\n\n"
        "2. /translate — выбери тайтл и главу\n"
        "   Бот скачает, переведёт и отправит PDF\n\n"
        "3. /list — все добавленные тайтлы\n\n"
        "Источник: MangaDex (корейский/английский)\n"
        "Перевод: LLM через Codespace (или Google Translate)"
    )
