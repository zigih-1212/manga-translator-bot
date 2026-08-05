import json
from aiogram import Router
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from sources.router import SourceRouter as MangaSourceRouter
from cfg import CONFIG, CONFIG_PATH

router = Router()
manga_router = MangaSourceRouter()


class AddTitleStates(StatesGroup):
    waiting_search = State()
    choosing_result = State()
    choosing_lang = State()


@router.message(Command("add_title"))
async def cmd_add_title(message: Message, state: FSMContext):
    await message.answer("Название тайтла (поиск на Mangakakalot):")
    await state.set_state(AddTitleStates.waiting_search)


class AddTitleStates(StatesGroup):
    waiting_search = State()
    choosing_result = State()
    choosing_lang = State()


@router.message(Command("add_title"))
async def cmd_add_title(message: Message, state: FSMContext):
    await message.answer("Название тайтл (поиск на Mangakakalot):")
    await state.set_state(AddTitleStates.waiting_search)


@router.message(AddTitleStates.waiting_search)
async def process_search(message: Message, state: FSMContext):
    query = message.text.strip()
    await message.answer(f"Ищу «{query}»...")

    try:
        results = await manga_router.search(query)
    except Exception as e:
        await message.answer(
            f"Ошибка поиска: {e}\n"
            f"Попробуй ещё раз (название) или напиши /cancel"
        )
        return
    if not results:
        await message.answer("Ничего не найдено. Попробуй другое название:")
        return

    # Show results without fetching chapters (faster)
    # Deduplicate by title, keep first occurrence
    seen = set()
    unique_results = []
    for r in results:
        key = r.title.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
        if len(unique_results) >= 5:
            break

    await state.update_data(search_results=[{"id": r.id, "title": r.title, "source": r.source} for r in unique_results])
    buttons = []
    for i, r in enumerate(unique_results):
        text = f"{r.title}"
        if r.status:
            text += f" [{r.status}]"
        src_tag = {"mangakakalot": "MK", "manganelo": "MN"}.get(r.source, r.source.upper())
        buttons.append([InlineKeyboardButton(text=f"[{src_tag}] {text}", callback_data=f"add:{i}")])
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.answer("Выбери тайтл:", reply_markup=kb)
    await state.set_state(AddTitleStates.choosing_result)
    await message.answer("Выбери тайтл:", reply_markup=kb)
    await state.set_state(AddTitleStates.choosing_result)


@router.callback_query(lambda c: c.data.startswith("add:"))
async def select_title(callback: CallbackQuery, state: FSMContext):
    idx = int(callback.data.split(":", 1)[1])
    data = await state.get_data()
    results = data.get("search_results", [])
    if idx >= len(results):
        await callback.message.answer("Ошибка: результат не найден")
        await callback.answer()
        await state.clear()
        return
    manga_id = results[idx]["id"]
    title_name = results[idx]["title"]
    source_name = results[idx].get("source", "mangakakalot")
    await state.update_data(manga_name=title_name, source=source_name)
    await callback.answer()

    await callback.message.answer("Проверяю доступные языки...")
    # Mangakakalot has English chapters, default to Korean original
    langs = ["ko", "en"]
    lang_list = "Korean (оригинал), English"

    await state.update_data(manga_id=manga_id, available_langs=langs)

    buttons = []
    for lang in langs:
        name_map = {"ko": "Korean (оригинал)", "en": "English", "ru": "Russian", "ja": "Japanese"}
        buttons.append([InlineKeyboardButton(
            text=name_map.get(lang, lang),
            callback_data=f"alang:{lang}"
        )])
    if not buttons:
        buttons = [[InlineKeyboardButton(text="OK", callback_data="alang:ko")]]
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)
    await callback.message.answer(
        f"Доступные языки: {lang_list}\n\n"
        f"Какой язык — оригинал (откуда переводить)?",
        reply_markup=kb,
    )
    await state.set_state(AddTitleStates.choosing_lang)


@router.callback_query(lambda c: c.data.startswith("alang:"))
async def select_source_lang(callback: CallbackQuery, state: FSMContext):
    source_lang = callback.data.split(":", 1)[1]
    await callback.answer()

    data = await state.get_data()
    manga_id = data.get("manga_id")
    source_name = data.get("source", "mangakakalot")
    if not manga_id:
        await callback.message.answer(
            "Сессия добавления тайтла сброшена (бот перезапускался). "
            "Начни заново: /add_title"
        )
        await state.clear()
        return
    title_name = data.get("manga_name", f"Mangakakalot:{manga_id[:8]}")
    await callback.message.answer(f"Ищу главы на языке «{source_lang}»...")
    source = await manga_router.get(source_name)
    chapters = await source.get_chapters(manga_id, source_lang)
    await source.close()

    if not chapters:
        await callback.message.answer(f"Нет глав на языке «{source_lang}».")
        await state.clear()
        return

    title_entry = {
        "name": title_name,
        "manga_id": manga_id,
        "source": source_name,
        "source_lang": source_lang,
        "chapters_count": len(chapters),
        "first_chapter": chapters[0].number if chapters else "",
        "last_chapter": chapters[-1].number if chapters else "",
    }

    existing = [t.get("manga_id") for t in CONFIG.get("titles", [])]
    if manga_id not in existing:
        CONFIG["titles"].append(title_entry)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    await callback.message.answer(
        f"Тайтл добавлен!\n\n"
        f"Mangakakalot ID: {manga_id}\n"
        f"Язык оригинала: {source_lang}\n"
        f"Глав: {len(chapters)}\n"
        f"Диапазон: {chapters[0].number} — {chapters[-1].number}\n\n"
        f"Используй /translate для перевода."
    )
    await state.clear()


async def _list_titles_impl(message: Message):
    """Показать список тайтлов (вызывается из команды и из меню)."""
    titles = CONFIG.get("titles", [])
    if not titles:
        await message.answer("📭 Нет добавленных тайтлов.\nИспользуй ➕ <b>Добавить тайтл</b>", parse_mode="HTML")
        return

    text = "📚 <b>Мои тайтлы:</b>\n\n"
    for i, t in enumerate(titles, 1):
        text += f"{i}. <b>{t['name']}</b>\n"
        text += f"   🌐 Язык: {t.get('source_lang', '?')} | 📖 Глав: {t.get('chapters_count', '?')}\n"
        text += f"   📌 Последняя: гл. {t.get('last_chapter', '?')}\n\n"
    await message.answer(text, parse_mode="HTML")


@router.message(Command("list"))
async def cmd_list(message: Message):
    await _list_titles_impl(message)
