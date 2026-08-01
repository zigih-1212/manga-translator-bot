import asyncio
import io
from pathlib import Path
from aiogram import Router
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.filters import Command
from aiogram.types import FSInputFile
from PIL import Image
import zipfile
from cfg import CONFIG, save_config
from translator.pipeline import TranslationPipeline
from cfg.db import TranslationQueueDB
from sources.mangadex import MangaDexSource

router = Router()
active_tasks: dict[int, asyncio.Task] = {}

db = TranslationQueueDB()


class TranslateStates(StatesGroup):
    choosing_title = State()
    choosing_chapter = State()


@router.message(Command("translate"))
async def cmd_translate(message: Message, state: FSMContext):
    titles = CONFIG.get("titles", [])
    if not titles:
        await message.answer("Нет тайтлов. Сначала /add_title")
        return

    buttons = []
    for i, t in enumerate(titles):
        buttons.append([InlineKeyboardButton(
            text=f"{t['name']} ({t.get('chapters_count', '?')} глав)",
            callback_data=f"tr_title:{i}"
        )])
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.answer("Выбери тайтл:", reply_markup=kb)
    await state.set_state(TranslateStates.choosing_title)


@router.callback_query(lambda c: c.data.startswith("tr_title:"))
async def select_title(callback: CallbackQuery, state: FSMContext):
    idx = int(callback.data.split(":")[1])
    title = CONFIG["titles"][idx]
    await state.update_data(title_idx=idx, title=title)
    await callback.message.answer(
        f"Тайтл: {title['name']}\n"
        f"Язык: {title.get('source_lang', '?')}\n"
        f"Главы: {title.get('first_chapter', '?')} — {title.get('last_chapter', '?')}\n\n"
        f"Номер главы?"
    )
    await callback.answer()
    await state.set_state(TranslateStates.choosing_chapter)


@router.message(TranslateStates.choosing_chapter)
async def process_chapter(message: Message, state: FSMContext):
    chapter_num = message.text.strip()
    data = await state.get_data()
    title = data["title"]

    manga_id = title["mangadex_id"]
    source_lang = title.get("source_lang", "ko")

    if db.add_to_queue(manga_id, chapter_num, source_lang):
        await message.answer(
            f"Глава {chapter_num} для тайтла {title['name']} добавлена в очередь перевода."
        )
    else:
        await message.answer(
            f"Глава {chapter_num} для тайтла {title['name']} уже в очереди или переводится."
        )
    await state.clear()


@router.callback_query(lambda c: c.data == "tr_cancel")
async def cancel_translate(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Отменено.")
    await callback.answer()
    await state.clear()


@router.callback_query(lambda c: c.data == "tr_start")
async def start_translate_now(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Перевод будет запущен автоматически из очереди.")
    await callback.answer()
    await state.clear()


# Новый обработчик для команды /translate_all
@router.message(Command("translate_all"))
async def cmd_translate_all(message: Message, state: FSMContext):
    titles = CONFIG.get("titles", [])
    if not titles:
        await message.answer("Нет тайтлов. Сначала /add_title")
        return

    added_count = 0
    for title_entry in titles:
        manga_id = title_entry["mangadex_id"]
        source_lang = title_entry.get("source_lang", "ko")

        mangadex_source = MangaDexSource()
        chapters = await mangadex_source.get_chapters(manga_id, source_lang)
        await mangadex_source.close()

        for chapter in chapters:
            if db.add_to_queue(manga_id, chapter.number, source_lang):
                added_count += 1
    
    await message.answer(f"Добавлено {added_count} глав в очередь перевода.")
    await state.clear()


# Добавляем команду /queue_status для отображения очереди
@router.message(Command("queue_status"))
async def cmd_queue_status(message: Message):
    tasks = db.get_all_tasks()
    if not tasks:
        await message.answer("Очередь перевода пуста.")
        return

    pending = [t for t in tasks if t["status"] == "pending"]
    processing = [t for t in tasks if t["status"] == "processing"]
    completed = [t for t in tasks if t["status"] == "completed"]
    failed = [t for t in tasks if t["status"] == "failed"]

    response_text = "<b>Статус очереди перевода:</b>\n\n"

    titles = CONFIG.get("titles", [])
    title_name = titles[0]["name"] if titles else "Манга"

    if processing:
        response_text += "<b>В работе:</b>\n"
        for t in processing:
            response_text += f"- {title_name} - гл. {t['chapter_number']} ({t['source_lang']}->ru)\n"
        response_text += "\n"

    if pending:
        response_text += "<b>Ожидают:</b>\n"
        for t in pending:
            response_text += f"- {title_name} - гл. {t['chapter_number']} ({t['source_lang']}->ru)\n"
        response_text += "\n"
    
    if failed:
        response_text += "<b>Ошибки:</b>\n"
        for t in failed:
            response_text += f"- {title_name} - гл. {t['chapter_number']} ({t['source_lang']}->ru) | Ошибка: {t['error_message']}\n"
        response_text += "\n"

    response_text += f"Всего задач: {len(tasks)} (Выполнено: {len(completed)}, Ожидают: {len(pending)}, В работе: {len(processing)}, Ошибки: {len(failed)})\n"

    await message.answer(response_text, parse_mode="HTML")


@router.message(Command("chapters"))
async def cmd_chapters(message: Message):
    """Show real chapter count from MangaDex for a selected title."""
    titles = CONFIG.get("titles", [])
    if not titles:
        await message.answer("Нет тайтлов. Сначала /add_title")
        return

    buttons = []
    for i, t in enumerate(titles):
        buttons.append([InlineKeyboardButton(
            text=f"{t['name']} ({t.get('chapters_count', '?')} глав)",
            callback_data=f"chapters_title:{i}",
        )])
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)
    await message.answer("Выбери тайтл для проверки глав:", reply_markup=kb)


@router.callback_query(lambda c: c.data.startswith("chapters_title:"))
async def check_chapters(callback: CallbackQuery):
    idx = int(callback.data.split(":")[1])
    title = CONFIG["titles"][idx]
    manga_id = title["mangadex_id"]
    source_lang = title.get("source_lang", "ko")

    await callback.answer("Проверяю на MangaDex...")
    await callback.message.answer(f"🔍 Проверяю главы «{title['name']}» (язык: {source_lang})...")

    try:
        mangadex_source = MangaDexSource()
        chapters = await mangadex_source.get_chapters(manga_id, source_lang)
        await mangadex_source.close()

        if not chapters:
            await callback.message.answer(f"Главы не найдены для «{title['name']}» на языке {source_lang}.")
            return

        numbers = sorted(set(float(c.number) for c in chapters if c.number))
        await callback.message.answer(
            f"📚 <b>{title['name']}</b>\n"
            f"🔗 MangaDex ID: <code>{manga_id}</code>\n"
            f"🌐 Язык: {source_lang}\n"
            f"📖 Найдено глав: <b>{len(chapters)}</b> (уникальных номеров: <b>{len(numbers)}</b>)\n"
            f"📐 Диапазон: <code>{numbers[0]:.0f}</code> — <code>{numbers[-1]:.0f}</code>\n"
            f"✅ В очереди: <b>{db.get_pending_count(manga_id)}</b> ожидают, "
            f"<b>{db.get_processing_count(manga_id)}</b> в работе",
            parse_mode="HTML",
        )
    except Exception as e:
        await callback.message.answer(f"Ошибка проверки глав: {e}")


@router.shutdown()
async def shutdown_handler():
    db.close()
