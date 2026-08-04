import asyncio
import aiohttp
import zipfile
from pathlib import Path
from aiogram import Router, F
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, FSInputFile
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from sources.router import SourceRouter
from translator.pipeline import TranslationPipeline
from translator.health import record_error
from translator.webhooks import notify_chapter_done, notify_chapter_failed
from cfg import CONFIG, save_config
from bot.utils.telegram_helpers import send_text, send_document

router = Router()
sources = SourceRouter()
TEMP_DIR = Path(__file__).resolve().parent.parent.parent / "temp"
active_tasks: dict[int, asyncio.Task] = {}


class MangaTranslateStates(StatesGroup):
    choosing = State()
    choosing_range = State()


@router.message(Command("manga"))
async def cmd_manga(message: Message, state: FSMContext):
    query = message.text.removeprefix("/manga").strip()
    if not query:
        await message.answer(
            "Напиши название после /manga, например:\n"
            "/manga Поднятие уровня в одиночку\n\n"
            "Покажу 3 варианта с обложками — выбери, введи диапазон глав (например 5-20) и я переведу их."
        )
        return
    await message.answer(f"🔍 Ищу «{query}»...")
    try:
        results = await sources.search(query)
    except Exception as e:
        await message.answer(f"Ошибка поиска: {e}. Попробуй позже.")
        return
    if not results:
        await message.answer("Ничего не найдено. Попробуй другое название.")
        return

    top3 = results[:3]
    await state.set_data({"search_results": top3})
    await state.set_state(MangaTranslateStates.choosing)

    buttons = []
    for i, r in enumerate(top3):
        label = r.title
        if not label or label == "Unknown":
            label = r.alt_titles[0] if r.alt_titles else "Unknown"
        src_tag = {"mangadex": "MD", "naver": "NAVER"}.get(r.source, r.source.upper())
        buttons.append([InlineKeyboardButton(
            text=f"✅ [{src_tag}] {label[:60]}",
            callback_data=f"mtr:{i}",
        )])
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)

    # Show covers inline (as media group) for the 3 candidates
    await message.answer(f"Нашёл {len(top3)} варианта. Выбери:", reply_markup=kb)

    for r in top3:
        caption = r.title if r.title != "Unknown" else (r.alt_titles[0] if r.alt_titles else "Unknown")
        if r.source:
            caption = f"[{r.source}] {caption}"
        if r.status:
            caption += f" [{r.status}]"
        if r.cover_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(r.cover_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        if resp.status == 200:
                            cover_bytes = await resp.read()
                TEMP_DIR.mkdir(parents=True, exist_ok=True)
                cover_path = TEMP_DIR / f"cover_{r.id}.jpg"
                cover_path.write_bytes(cover_bytes)
                await message.answer_photo(FSInputFile(str(cover_path)), caption=caption)
                cover_path.unlink(missing_ok=True)
                continue
            except Exception:
                pass
        await message.answer(caption)


@router.callback_query(MangaTranslateStates.choosing, F.data.startswith("mtr:"))
async def select_manga(callback: CallbackQuery, state: FSMContext):
    idx = int(callback.data.split(":", 1)[1])
    data = await state.get_data()
    results = data.get("search_results", [])
    if idx >= len(results):
        await callback.message.answer("Ошибка выбора.")
        await callback.answer()
        await state.clear()
        return
    r = results[idx]
    lang_map = {"ko": "ko", "ja": "ja", "zh": "zh", "en": "en"}
    source_lang = lang_map.get(r.original_language, "ko")
    await state.update_data(selected_manga=r, source_lang=source_lang, source=r.source)
    await callback.message.answer(
        f"✅ Выбрано: <b>{r.title}</b>\n"
        f"🌐 Язык оригинала: {r.original_language}\n\n"
        f"Введи диапазон глав (например <code>5-20</code>) или <code>1</code> для одной главы.\n"
        f"Бот переведёт их по очереди и отправит ZIP после каждой.",
        parse_mode="HTML",
    )
    await callback.answer()
    await state.set_state(MangaTranslateStates.choosing_range)


@router.message(MangaTranslateStates.choosing_range)
async def process_range(message: Message, state: FSMContext):
    text = message.text.strip()
    data = await state.get_data()
    r = data.get("selected_manga")
    if not r:
        await message.answer("Сессия устарела. Начни заново: /manga <название>")
        await state.clear()
        return
    source_lang = data.get("source_lang", "ko")

    # Parse "5-20" or single "5"
    if "-" in text:
        parts = text.replace(" ", "").split("-")
        if len(parts) != 2:
            await message.answer("Неверный формат. Пример: <code>5-20</code>", parse_mode="HTML")
            return
        try:
            start_c, end_c = int(parts[0]), int(parts[1])
        except ValueError:
            await message.answer("Нужны числа. Пример: <code>5-20</code>", parse_mode="HTML")
            return
        if start_c < 1 or end_c < start_c:
            await message.answer("Диапазон некорректен (например 5-20).", parse_mode="HTML")
            return
        chapter_nums = [str(n) for n in range(start_c, end_c + 1)]
    else:
        try:
            chapter_nums = [str(int(text))]
        except ValueError:
            await message.answer("Нужно число или диапазон. Пример: <code>5-20</code>", parse_mode="HTML")
            return

    chat_id = message.chat.id
    source = data.get("source", "mangadex")
    await state.clear()

    if chat_id in active_tasks and not active_tasks[chat_id].done():
        await message.answer("Перевод уже запущен. Дождись завершения.")
        return

    task = asyncio.create_task(run_translation(chat_id, r, chapter_nums, source_lang, source))
    active_tasks[chat_id] = task
    await message.answer(
        f"🚀 Начинаю перевод <b>{r.title}</b>:\n"
        f"Главы: {chapter_nums[0]}–{chapter_nums[-1]} ({len(chapter_nums)} шт.)\n"
        f"ZIP будет отправлен после каждой главы.",
        parse_mode="HTML",
    )


async def run_translation(chat_id: int, r, chapter_nums: list[str], source_lang: str, source: str = "mangadex"):
    manga_id = r.id
    try:
        for i, ch_num in enumerate(chapter_nums, 1):
            try:
                await send_text(chat_id, f"📄 [{i}/{len(chapter_nums)}] Глава {ch_num}: перевод...")
                pipeline = TranslationPipeline(source=source)
                page_paths = await pipeline.process_chapter(
                    mangadex_manga_id=manga_id,
                    chapter_number=ch_num,
                    source_lang=source_lang,
                    target_lang="en",
                    source=source,
                )
                await pipeline.close()

                if not page_paths:
                    await send_text(chat_id, f"❌ Глава {ch_num}: не удалось перевести (глава не найдена).")
                    await notify_chapter_failed(r.title, ch_num, "глава не найдена")
                    continue

                zip_path = TEMP_DIR / f"manga_{manga_id[:8]}_ch_{ch_num}.zip"
                zip_path.parent.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in page_paths:
                        zf.write(p, arcname=Path(p).name)

                await send_document(
                    chat_id, zip_path,
                    caption=f"{r.title} — глава {ch_num}",
                )
                zip_path.unlink(missing_ok=True)
                await send_text(chat_id, f"✅ Глава {ch_num} готова!")
                await notify_chapter_done(r.title, ch_num)
            except Exception as e:
                record_error()
                await send_text(chat_id, f"❌ Глава {ch_num}: ошибка: {e}")
                await notify_chapter_failed(r.title, ch_num, str(e))

            if i < len(chapter_nums):
                await asyncio.sleep(10)  # delay between chapters

        await send_text(chat_id, f"🏁 Завершено: {len(chapter_nums)} глав обработано.")
    finally:
        active_tasks.pop(chat_id, None)
