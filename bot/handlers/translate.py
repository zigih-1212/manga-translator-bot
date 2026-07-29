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
import img2pdf
from config import CONFIG
from translator.pipeline import TranslationPipeline

router = Router()
active_tasks: dict[int, asyncio.Task] = {}


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

    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=f"Начать перевод главы {chapter_num}", callback_data="tr_start")],
        [InlineKeyboardButton(text="Отмена", callback_data="tr_cancel")],
    ])
    await message.answer(
        f"Тайтл: {title['name']}\n"
        f"Глава: {chapter_num}\n"
        f"Язык оригинала: {title.get('source_lang', 'ko')}\n\n"
        f"Начать?",
        reply_markup=kb,
    )
    await state.update_data(chapter_number=chapter_num)


@router.callback_query(lambda c: c.data == "tr_cancel")
async def cancel_translate(callback: CallbackQuery, state: FSMContext):
    await callback.message.answer("Отменено.")
    await callback.answer()
    await state.clear()


@router.callback_query(lambda c: c.data == "tr_start")
async def start_translate(callback: CallbackQuery, state: FSMContext):
    data = await state.get_data()
    title = data["title"]
    chapter_number = data["chapter_number"]
    user_id = callback.from_user.id
    await callback.answer()

    if user_id in active_tasks and not active_tasks[user_id].done():
        await callback.message.answer("Уже идёт перевод. Дождись.")
        return

    await state.clear()

    progress_msg = await callback.message.answer("Подготовка...")

    pipeline = TranslationPipeline()
    source_lang = title.get("source_lang", "ko")

    async def update_progress(msg: str, current: int, total: int):
        bar_len = 20
        filled = int(bar_len * current / total) if total else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        try:
            await progress_msg.edit_text(
                f"Перевод: {title['name']} — стр. {chapter_number}\n\n"
                f"{bar} {current}%\n\n{msg}"
            )
        except Exception:
            pass

    pipeline.on_progress(update_progress)

    async def run_translation():
        try:
            page_paths = await pipeline.process_chapter(
                mangadex_manga_id=title["mangadex_id"],
                chapter_number=chapter_number,
                source_lang=source_lang,
                target_lang="en",
            )

            if not page_paths:
                await progress_msg.edit_text("Не удалось перевести главу.")
                return

            await progress_msg.edit_text(
                f"Собираю PDF ({len(page_paths)} стр.)..."
            )

            pdf_path = Path("temp") / f"chapter_{chapter_number}.pdf"
            pdf_path.parent.mkdir(parents=True, exist_ok=True)

            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert(page_paths))

            total_pages_pdf = len(page_paths)

            await callback.message.answer_document(
                document=FSInputFile(str(pdf_path)),
                caption=(
                    f"{title['name']} — глава {chapter_number}\n"
                    f"Страниц: {total_pages_pdf}\n"
                    f"Язык: {source_lang} -> ru"
                ),
            )

            await progress_msg.edit_text(
                f"Готово!\n"
                f"Тайтл: {title['name']}\n"
                f"Глава: {chapter_number}\n"
                f"Страниц: {total_pages_pdf}"
            )

        except Exception as e:
            await progress_msg.edit_text(f"Ошибка: {e}")
        finally:
            active_tasks.pop(user_id, None)
            await pipeline.close()

    task = asyncio.create_task(run_translation())
    active_tasks[user_id] = task
