from aiogram import Router
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
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
async def cmd_start(message: Message, state: FSMContext):
    await state.clear()
    CONFIG.setdefault("telegram", {})["chat_id"] = message.chat.id
    save_config()
    await message.answer(
        "👋 Привет! Я бот для перевода манги с Mangakakalot.\n"
        "Выбери действие ниже:",
        reply_markup=build_main_menu()
    )


@router.message(Command("help"))
async def cmd_help(message: Message):
    await message.answer(
        "📖 <b>Как пользоваться:</b>\n\n"
        "1️⃣ <b>Добавить тайтл</b> — поиск на Mangakakalot по названию\n"
        "2️⃣ <b>Поиск и перевод</b> — быстрый поиск + выбор глав\n"
        "3️⃣ <b>Мои тайтлы</b> — список добавленных, авто-проверка новых глав\n"
        "4️⃣ <b>Статус очереди</b> — что переводится прямо сейчас\n\n"
        "⚡ Бот каждые 6 часов сам проверяет новые главы.\n"
        "📦 Готовые главы приходят ZIP-архивом.",
        parse_mode="HTML"
    )


@router.callback_query(lambda c: c.data == "menu:add_title")
async def cb_add_title(callback: CallbackQuery, state: FSMContext):
    """Кнопка «Добавить тайтл» — запускает поиск."""
    await state.clear()
    await callback.message.answer("🔍 Название тайтла (поиск на Mangakakalot):")
    from bot.handlers.titles import AddTitleStates
    await state.set_state(AddTitleStates.waiting_search)
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:search_translate")
async def cb_search_translate(callback: CallbackQuery):
    """Кнопка «Поиск и перевод» — запуск поиска."""
    await callback.message.answer(
        "🔍 Напиши название манги для поиска на Mangakakalot.\n"
        "Покажу 3 варианта с обложками — выбери и введи диапазон глав.\n\n"
        "Например: <code>/manga Solo Leveling</code>",
        parse_mode="HTML"
    )
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:list_titles")
async def cb_list_titles(callback: CallbackQuery):
    """Кнопка «Мои тайтлы» — показать список."""
    from bot.handlers.titles import _list_titles_impl
    await _list_titles_impl(callback.message)
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:status")
async def cb_status(callback: CallbackQuery):
    """Кнопка «Статус очереди» — показать очередь."""
    from bot.handlers.status import _status_impl
    await _status_impl(callback.message)
    await callback.answer()


@router.callback_query(lambda c: c.data == "menu:help")
async def cb_help(callback: CallbackQuery):
    """Кнопка «Помощь» — редактировать сообщение с меню."""
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
