from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command

router = Router()


async def _status_impl(message: Message):
    """Показать статус очереди (вызывается из команды и из меню)."""
    from bot.handlers.translate import active_tasks
    from cfg.db import TranslationQueueDB

    user_id = message.from_user.id
    lines = []

    # Проверяем активные задачи из хендлеров
    if user_id in active_tasks and not active_tasks[user_id].done():
        lines.append("🔄 Перевод в процессе (запущен вручную)...")

    db = TranslationQueueDB()
    tasks = db.get_all_tasks()
    db.close()

    if tasks:
        pending = [t for t in tasks if t["status"] == "pending"]
        processing = [t for t in tasks if t["status"] == "processing"]
        completed = [t for t in tasks if t["status"] == "completed"]
        failed = [t for t in tasks if t["status"] == "failed"]
        lines.append("")
        for t in processing:
            lines.append(f"🔧 <code>{t['manga_id'][:8]}</code> — гл. {t['chapter_number']}: в работе")
        for t in pending:
            lines.append(f"⏳ <code>{t['manga_id'][:8]}</code> — гл. {t['chapter_number']}: ожидает")
        for t in failed:
            lines.append(f"❌ <code>{t['manga_id'][:8]}</code> — гл. {t['chapter_number']}: ошибка")
        lines.append("")
        lines.append(
            f"📊 <b>Итого:</b> {len(tasks)} (✅ {len(completed)}, "
            f"🔧 {len(processing)}, ⏳ {len(pending)}, ❌ {len(failed)})"
        )
    elif not lines:
        lines.append("📭 Нет активных задач.")

    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("status"))
async def cmd_status(message: Message):
    await _status_impl(message)
