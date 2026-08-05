from aiogram import Router
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import CommandStart, Command
from cfg import CONFIG, save_config

router = Router()


def build_main_menu() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(text="➕ Добавить тайтл", callback_data="menu:add_title")],
        [InlineKeyboardButton(text="🔍 Поиск и перевод", callback_data="menu:search_translate")],
        [InlineKeyboardButton(text="📋 Мои тайтлы", callback_data="menu:list_titles")],
        [InlineKeyboardButton(text="📊 Статус очереди", callback_data="menu:status")],
        [InlineKeyboardButton(text="❓ Помощь", callback_data="menu:help")],
    ]
    return InlineKeyboardMarkup(inline_keyboard=buttons)


@router.message(CommandStart())
async def cmd_start(message: Message):
    CONFIG.setdefault("telegram", {})["chat_id"] = message.chat.id
    save_config()
    await message.answer(
        "👋 Привет! Я бот для перевода манги с MangaDex.\n"
        "Выбери действие ниже:",
        reply_markup=build_main_menu()
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "📖 <b>Как пользоваться:</b>\n\n"
        "1️⃣ <b>Добавить тайтл</b> — поиск на MangaDex по названию\n"
        "2️⃣ <b>Поиск и перевод</b> — быстрый поиск + выбор глав\n"
        "3️⃣ <b>Мои тайтлы</b> — список добавленных, авто-проверка новых глав\n"
        "4️⃣ <b>Статус очереди</b> — что переводится прямо сейчас\n\n"
        "⚡ Бот каждые 6 часов сам проверяет новые главы.\n"
        "📦 Готовые главы приходят ZIP-архивом.",
        parse_mode="HTML"
    )


@router.callback_query(lambda c: c.data == "menu:add_title")
async def cb_add_title(callback):
    from bot.handlers.titles import cmd_add_title
    # Create a fake message to reuse the handler
    class FakeMsg:
        def __init__(self, original):
            self.chat = original.message.chat
            self.from_user = original.from_user
            self.text = "/add_title"
    await cmd_add_title(FakeMsg(callback))
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:search_translate")
async def cb_search_translate(callback):
    from bot.handlers.manga_translate import cmd_manga
    class FakeMsg:
        def __init__(self, original):
            self.chat = original.message.chat
            self.from_user = original.from_user
            self.text = "/manga "
    await cmd_manga(FakeMsg(callback))
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:list_titles")
async def cb_list_titles(callback):
    from bot.handlers.titles import cmd_list_titles
    class FakeMsg:
        def __init__(self, original):
            self.chat = original.message.chat
            self.from_user = original.from_user
    await cmd_list_titles(FakeMsg(callback))
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:status")
async def cb_status(callback):
    from bot.handlers.status import cmd_status
    class FakeMsg:
        def __init__(self, original):
            self.chat = original.message.chat
            self.from_user = original.from_user
    await cmd_status(FakeMsg(callback))
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:help")
async def cb_help(callback):
    await callback.message.edit_text(
        "📖 <b>Как пользоваться:</b>\n\n"
        "1️⃣ <b>Добавить тайтл</b> — поиск на MangaDex по названию\n"
        "2️⃣ <b>Поиск и перевод</b> — быстрый поиск + выбор глав\n"
        "3️⃣ <b>Мои тайтлы</b> — список добавленных, авто-проверка новых глав\n"
        "4️⃣ <b>Статус очереди</b> — что переводится прямо сейчас\n\n"
        "⚡ Бот каждые 6 часов сам проверяет новые главы.\n"
        "📦 Готовые главы приходят ZIP-архивом.",
        parse_mode="HTML",
        reply_markup=build_main_menu()
    )
    await callback.answer()
