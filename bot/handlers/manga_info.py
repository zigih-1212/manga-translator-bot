import aiohttp
from aiogram import Router, F
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery, FSInputFile
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from pathlib import Path
from sources.mangadex import MangaDexSource

router = Router()
mangadex = MangaDexSource()
TEMP_DIR = Path(__file__).resolve().parent.parent.parent / "temp"


class MangaInfoStates(StatesGroup):
    choosing = State()


@router.message(Command("manga"))
async def cmd_manga(message: Message, state: FSMContext):
    query = message.text.removeprefix("/manga").strip()
    if not query:
        await message.answer("Напиши название после /manga, например:\n/manga One Piece")
        return
    await message.answer(f"Ищу «{query}»...")
    results = await mangadex.search(query)
    if not results:
        await message.answer("Ничего не найдено. Попробуй другое название.")
        return
    buttons = []
    for i, r in enumerate(results[:5]):
        label = r.title
        if not label or label == "Unknown":
            label = r.alt_titles[0] if r.alt_titles else "Unknown"
        if r.status:
            label += f" [{r.status}]"
        buttons.append([InlineKeyboardButton(text=label[:60], callback_data=f"minfo:{i}")])
    kb = InlineKeyboardMarkup(inline_keyboard=buttons)
    await state.set_data({"search_results": results[:5]})
    await state.set_state(MangaInfoStates.choosing)
    await message.answer("Нашёл несколько. Выбери:", reply_markup=kb)


@router.callback_query(MangaInfoStates.choosing, F.data.startswith("minfo:"))
async def select_manga_info(callback: CallbackQuery, state: FSMContext):
    idx = int(callback.data.split(":", 1)[1])
    data = await state.get_data()
    results = data.get("search_results", [])
    if idx >= len(results):
        await callback.message.answer("Ошибка")
        await callback.answer()
        await state.clear()
        return
    r = results[idx]
    await callback.answer()
    await state.clear()
    text = f"<b>{r.title}</b>\n"
    if r.alt_titles:
        text += f"📖 {', '.join(r.alt_titles[:3])}\n"
    if r.description:
        desc = r.description[:500]
        if len(r.description) > 500:
            desc += "..."
        text += f"\n{desc}\n"
    text += f"\n📊 Статус: {r.status or 'неизвестно'}"
    if r.year:
        text += f" | Год: {r.year}"
    text += f"\n🔗 mangadex.org/title/{r.id}"
    if r.cover_url:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(r.cover_url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        cover_bytes = await resp.read()
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            cover_path = TEMP_DIR / f"cover_{r.id}.jpg"
            cover_path.write_bytes(cover_bytes)
            await callback.message.answer_photo(
                FSInputFile(str(cover_path)), caption=text, parse_mode="HTML"
            )
            cover_path.unlink(missing_ok=True)
            return
        except Exception:
            pass
    await callback.message.answer(text, parse_mode="HTML")
