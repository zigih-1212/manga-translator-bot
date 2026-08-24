from aiogram import Router
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from sources.router import SourceRouter as MangaSourceRouter
from cfg import CONFIG

router = Router()
manga_router = MangaSourceRouter()


class AddTitleStates(StatesGroup):
    waiting_search = State()
    choosing_result = State()


@router.message(Command("add_title"))
async def cmd_add_title(message: Message, state: FSMContext):
    await message.answer("🔍 Название тайтла (поиск в английских источниках):")
    await state.set_state(AddTitleStates.waiting_search)


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

    await callback.message.answer(f"📖 Загружаю список глав «{title_name}»...")
    source = await manga_router.get(source_name)
    # English aggregator sources serve EN chapters; try requested langs in order.
    chapters: list = []
    used_lang = "en"
    for lang_try in ("en", "ko"):
        chapters = await source.get_chapters(manga_id, lang_try)
        if chapters:
            used_lang = lang_try
            break
    await source.close()

    if not chapters:
        await callback.message.answer(
            f"❌ Главы не найдены для «{title_name}» ({source_name}).\n"
            f"Попробуй другой вариант из поиска."
        )
        await state.clear()
        return

    nums = sorted(chapters, key=lambda c: float(c.number) if c.number.replace(".", "", 1).isdigit() else 0)
    title_entry = {
        "name": title_name,
        "manga_id": manga_id,
        "source": source_name,
        "source_lang": used_lang,
        "chapters_count": len(nums),
        "first_chapter": nums[0].number,
        "last_chapter": nums[-1].number,
    }

    existing = [t.get("manga_id") for t in CONFIG.get("titles", [])]
    if manga_id not in existing:
        CONFIG["titles"].append(title_entry)
        from bot.utils.ops import asave_config
        await asave_config()

    await callback.message.answer(
        f"✅ <b>Тайтл добавлен!</b>\n\n"
        f"📖 {title_name}\n"
        f"🌐 Источник: {source_name} (язык: {used_lang})\n"
        f"📚 Глав: {len(nums)}  (диапазон {nums[0].number}–{nums[-1].number})\n\n"
        f"Перевод: /translate или /manga",
        parse_mode="HTML",
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
