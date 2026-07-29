import json
from aiogram import Router
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from sources.mangadex import MangaDexSource
from config import CONFIG, CONFIG_PATH

router = Router()
mangadex = MangaDexSource()


class AddTitleStates(StatesGroup):
    waiting_search = State()
    choosing_result = State()
    choosing_lang = State()


@router.message(Command("add_title"))
async def cmd_add_title(message: Message, state: FSMContext):
    await message.answer("Название тайтла (поиск на MangaDex):")
    await state.set_state(AddTitleStates.waiting_search)


@router.message(AddTitleStates.waiting_search)
async def process_search(message: Message, state: FSMContext):
    query = message.text.strip()
    await message.answer(f"Ищу «{query}» на MangaDex...")

    results = await mangadex.search(query)
    if not results:
        await message.answer("Ничего не найдено. Попробуй другое название:")
        return

    await state.update_data(search_results=[{"id": r.id, "title": r.title} for r in results[:5]])
    buttons = []
    for i, r in enumerate(results[:5]):
        text = f"{r.title}"
        if r.status:
            text += f" [{r.status}]"
        buttons.append([InlineKeyboardButton(text=text, callback_data=f"add:{i}")])
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)
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
    await state.update_data(manga_name=title_name)
    await callback.answer()

    await callback.message.answer("Проверяю доступные языки...")
    langs = await mangadex.get_available_languages(manga_id)
    lang_list = ", ".join(langs) if langs else "неизвестно"

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
    manga_id = data["manga_id"]
    title_name = data.get("manga_name", f"MangaDex:{manga_id[:8]}")
    await callback.message.answer(f"Ищу главы на языке «{source_lang}»...")
    chapters = await mangadex.get_chapters(manga_id, source_lang)

    if not chapters:
        await callback.message.answer(f"Нет глав на языке «{source_lang}».")
        await state.clear()
        return

    title_entry = {
        "name": title_name,
        "mangadex_id": manga_id,
        "source_lang": source_lang,
        "chapters_count": len(chapters),
        "first_chapter": chapters[0].number if chapters else "",
        "last_chapter": chapters[-1].number if chapters else "",
    }

    existing = [t["mangadex_id"] for t in CONFIG.get("titles", [])]
    if manga_id not in existing:
        CONFIG["titles"].append(title_entry)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, ensure_ascii=False, indent=2)

    await callback.message.answer(
        f"Тайтл добавлен!\n\n"
        f"MangaDex ID: {manga_id}\n"
        f"Язык оригинала: {source_lang}\n"
        f"Глав: {len(chapters)}\n"
        f"Диапазон: {chapters[0].number} — {chapters[-1].number}\n\n"
        f"Используй /translate для перевода."
    )
    await state.clear()


@router.message(Command("list"))
async def cmd_list(message: Message):
    titles = CONFIG.get("titles", [])
    if not titles:
        await message.answer("Нет добавленных тайтлов. /add_title")
        return

    text = "Тайтлы:\n\n"
    for i, t in enumerate(titles, 1):
        text += f"{i}. {t['name']}\n"
        text += f"   MangaDex: {t.get('mangadex_id', 'нет')}\n"
        text += f"   Язык: {t.get('source_lang', '?')}\n"
        text += f"   Глав: {t.get('chapters_count', '?')} ({t.get('first_chapter', '?')}—{t.get('last_chapter', '?')})\n\n"
    await message.answer(text)
