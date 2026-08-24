"""Control commands: /cancel, /retry, /stats, /check — queue & task management."""
import asyncio
import time
from aiogram import Router
from aiogram.types import Message
from aiogram.filters import Command

from cfg import CONFIG
from cfg.db import TranslationQueueDB

router = Router()

# Shared registry of running translation tasks (set by handlers/main loops)
RUNNING_TASKS: dict[str, asyncio.Task] = {}

# Simple in-process stats collector
_STATS = {
    "started_at": time.time(),
    "chapters_done": 0,
    "chapters_failed": 0,
    "pages_done": 0,
    "seconds_spent": 0.0,
    "per_chapter": [],  # [(ts, chapter_key, seconds, pages)]
}


def record_chapter_done(key: str, seconds: float, pages: int):
    _STATS["chapters_done"] += 1
    _STATS["seconds_spent"] += seconds
    _STATS["pages_done"] += pages
    _STATS["per_chapter"].append((time.time(), key, round(seconds, 1), pages))
    _STATS["per_chapter"] = _STATS["per_chapter"][-200:]  # keep last 200


def record_chapter_failed():
    _STATS["chapters_failed"] += 1


def cancel_task_for(manga_id: str, chapter: str) -> bool:
    """Cancel a running translation task by (manga_id, chapter)."""
    for key, task in list(RUNNING_TASKS.items()):
        if manga_id in key and (not chapter or chapter in key):
            if not task.done():
                task.cancel()
            RUNNING_TASKS.pop(key, None)
            return True
    return False


@router.message(Command("cancel"))
async def cmd_cancel(message: Message):
    """/cancel            -> stop current manual translation + clear pending queue
       /cancel 5         -> remove chapter 5 from queue / stop its task"""
    args = message.text.split(maxsplit=1)
    db = TranslationQueueDB()
    stopped = []
    if len(args) > 1:
        target = args[1].strip()
        # Cancel matching running tasks
        for key in list(RUNNING_TASKS):
            if target in key and not RUNNING_TASKS[key].done():
                RUNNING_TASKS[key].cancel()
                stopped.append(key)
        # Remove from DB queue
        removed = 0
        for t in db.get_pending_tasks():
            if t["chapter_number"] == target or target in t["manga_id"]:
                db.update_task_status(t["manga_id"], t["chapter_number"], "cancelled")
                removed += 1
        db.close()
        parts = []
        if stopped:
            parts.append(f"остановлено задач: {len(stopped)}")
        if removed:
            parts.append(f"убрано из очереди: {removed}")
        await message.answer("🛑 " + ("; ".join(parts) if parts else "ничего подходящего не найдено"))
        return

    # Global cancel: everything running + whole pending queue
    for key, task in list(RUNNING_TASKS.items()):
        if not task.done():
            task.cancel()
        RUNNING_TASKS.pop(key, None)
    n_pending = 0
    for t in db.get_pending_tasks():
        db.update_task_status(t["manga_id"], t["chapter_number"], "cancelled")
        n_pending += 1
    db.close()
    await message.answer(f"🛑 Остановлено задач: {len(RUNNING_TASKS)}, отменено в очереди: {n_pending}")


@router.message(Command("retry"))
async def cmd_retry(message: Message):
    """/retry          -> requeue ALL failed chapters
       /retry <num>   -> requeue failed chapter <num> of first title"""
    args = message.text.split(maxsplit=1)
    db = TranslationQueueDB()
    titles = CONFIG.get("titles", [])
    default_id = (titles[0].get("manga_id") if titles else None)

    if len(args) > 1:
        num = args[1].strip()
        requeued = 0
        for t in db.get_all_tasks():
            if t["status"] == "failed" and (
                t["chapter_number"] == num or (default_id and t["manga_id"] == default_id)
            ):
                db.update_task_status(t["manga_id"], t["chapter_number"], "pending")
                requeued += 1
        db.close()
        await message.answer(f"♻️ Перезапущено глав: {requeued}")
        return

    requeued = 0
    for t in db.get_all_tasks():
        if t["status"] == "failed":
            db.update_task_status(t["manga_id"], t["chapter_number"], "pending")
            requeued += 1
    db.close()
    await message.answer(f"♻️ Все упавшие главы перезапущены: {requeued}")


@router.message(Command("stats"))
async def cmd_stats(message: Message):
    """/stats — сводка по переводам и скорости."""
    db = TranslationQueueDB()
    tasks = db.get_all_tasks()
    db.close()

    done = sum(1 for t in tasks if t["status"] == "completed")
    failed = sum(1 for t in tasks if t["status"] == "failed")
    pending = sum(1 for t in tasks if t["status"] == "pending")
    processing = sum(1 for t in tasks if t["status"] == "processing")

    uptime = time.time() - _STATS["started_at"]
    h, rem = divmod(int(uptime), 3600)
    m, s = divmod(rem, 60)

    avg = (_STATS["seconds_spent"] / _STATS["chapters_done"]) if _STATS["chapters_done"] else 0
    recent = _STATS["per_chapter"][-5:]

    lines = [
        "📊 <b>Статистика</b>",
        "",
        f"⏱ Uptime: {h}ч {m}м {s}с",
        f"✅ Глав переведено (сессия): {_STATS['chapters_done']}",
        f"❌ Ошибок (сессия): {_STATS['chapters_failed']}",
        f"📄 Страниц обработано: {_STATS['pages_done']}",
    ]
    if avg:
        lines.append(f"⚡ Среднее время на главу: {avg:.0f}с ({avg/60:.1f} мин)")
    if recent:
        lines.append("")
        lines.append("Последние главы:")
        for ts, key, secs, pages in reversed(recent):
            lines.append(f"  • {key} — {secs}с, {pages} стр.")
    lines += [
        "",
        f"🗄 Очередь всего: {len(tasks)} (✅{done} ⏳{pending} 🔧{processing} ❌{failed})",
    ]
    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("check"))
async def cmd_check(message: Message):
    """/check <manga_id|название> <глава> — dry-run: есть ли глава и сколько страниц,
    БЕЗ перевода. Полезно перед запуском длинного диапазона."""
    from sources.router import SourceRouter
    parts = message.text.split()
    titles = CONFIG.get("titles", [])
    if len(parts) < 2:
        if not titles:
            await message.answer("Использование: /check <id> <глава>  (или добавьте тайтл сначала)")
            return
        t0 = titles[0]
        await message.answer(
            f"Использование: /check [{t0.get('name')}] <глава>\n"
            f"Тайтл по умолчанию: {t0.get('name')} ({t0.get('source')}) id={str(t0.get('manga_id'))[:16]}…"
        )
        return

    # resolve target title by index or fall back to first
    try:
        idx = int(parts[1])
        t = titles[idx]
        chapter = parts[2] if len(parts) > 2 else None
    except (ValueError, IndexError):
        t = titles[0] if titles else None
        chapter = parts[1]

    if not t or not chapter:
        await message.answer("Укажите номер главы: /check 5")
        return

    src_name = t.get("source", "mangakakalot")
    manga_id = t.get("manga_id")
    await message.answer(f"🔎 Проверяю «{t.get('name')}» гл. {chapter} через {src_name}…")
    router_ = SourceRouter()
    try:
        src = await router_.get(src_name)
        ch = None
        if hasattr(src, "find_chapter_by_number"):
            ch = await src.find_chapter_by_number(manga_id, chapter, t.get("source_lang", "en"))
        if ch is None:
            chapters = await src.get_chapters(manga_id)
            want = chapter
            for c in chapters:
                if c.number == want:
                    ch = c
                    break
        if ch is None:
            await message.answer(f"❌ Глава {chapter} не найдена на {src_name}.")
            return
        pages = await src.get_pages(ch.id)
        await src.close()
        await message.answer(
            f"✅ Глава {ch.number} найдена ({src_name})\n"
            f"📄 Страниц: {len(pages)}\n"
            f"🆔 chapter_id: {ch.id[:24]}"
        )
    except Exception as e:
        await message.answer(f"❌ Ошибка проверки: {e}")
