import asyncio
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
import httpx
import zipfile
from aiohttp import ClientSession
from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import FSInputFile
from cfg import TG_BOT_TOKEN, TG_PROXY_URL, REMOTE_SERVER_URL, CONFIG, save_config, validate_config
from bot.middleware import CommandResetState
from bot.utils.progress import get_reporter, clear_reporter
from bot.handlers import get_routers
from sources.router import SourceRouter
from translator.pipeline import TranslationPipeline
from translator.health import start_health_server, stop_health_server, mark_bot_started, record_error
from cfg.db import TranslationQueueDB

bot: Bot | None = None

HTTP_PROXY = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL = 600


async def _keepalive():
    if not REMOTE_SERVER_URL:
        return
    proxy = HTTP_PROXY or None
    while True:
        try:
            async with httpx.AsyncClient(proxy=proxy, timeout=10) as c:
                r = await c.get(f"{REMOTE_SERVER_URL}/health")
                logger.info(f"Keepalive ping: {r.status_code}")
        except Exception as e:
            logger.warning(f"Keepalive failed: {e}")
        await asyncio.sleep(KEEPALIVE_INTERVAL)

PROXY_FALLBACKS = [
    "socks5://37.18.73.60:5566",
    "socks5://81.177.165.209:10808",
]


class Socks5Session(AiohttpSession):
    def __init__(self, proxy_url: str):
        super().__init__()
        self._proxy_url = proxy_url

    async def create_session(self) -> ClientSession:
        from aiohttp_socks import ProxyConnector
        if self._session is None or self._should_reset_connector:
            if self._session and not self._session.closed:
                await self._session.close()
            connector = ProxyConnector.from_url(self._proxy_url)
            self._session = ClientSession(connector=connector)
            self._should_reset_connector = False
        return self._session


def build_session():
    if TG_PROXY_URL and TG_PROXY_URL.startswith("socks"):
        return Socks5Session(TG_PROXY_URL)
    if TG_PROXY_URL:
        return AiohttpSession(proxy=TG_PROXY_URL)
    return AiohttpSession()


async def check_new_chapters(bot: Bot):
    titles = CONFIG.get("titles", [])
    chat_id = CONFIG.get("telegram", {}).get("chat_id")
    if not chat_id:
        logger.warning("auto: нет chat_id — отправь /start")
        return

    router = SourceRouter()

    for title in titles:
        manga_id = title.get("mangadex_id") or title.get("manga_id")
        source_lang = title.get("source_lang", "ko")
        last_chapter = title.get("last_chapter", "0")
        source_name = title.get("source", "mangakakalot")

        try:
            source = await router.get(source_name)
            chapters = await source.get_chapters(manga_id, source_lang)

            new_chapters = []
            for ch in chapters:
                try:
                    if float(ch.number) > float(last_chapter):
                        new_chapters.append(ch)
                except ValueError:
                    continue

            if not new_chapters:
                await source.close()
                continue

            # Sort by number ascending
            new_chapters.sort(key=lambda c: float(c.number))

            for ch in new_chapters:
                ch_str = ch.number

                try:
                    reporter = get_reporter(chat_id, manga_id, ch_str)
                    await reporter.start(f"📖 {title['name']} — гл. {ch_str}: инициализация...")
                    pipeline = TranslationPipeline()

                    async def _on_progress(message: str, current: int, total: int):
                        stage_emoji = "🔄"
                        manga_name = title['name']
                        if any(k in message for k in ["Поиск", "find_chapter", "get_pages"]):
                            stage_emoji = "🌐"
                        elif any(k in message for k in ["page", "страница", "bubble", "bubble_clean", "Preprocess"]):
                            stage_emoji = "📄"
                        elif any(k in message for k in ["Перевод", "translate", "LLM"]):
                            stage_emoji = "🌐"
                        elif any(k in message for k in ["Сохранено", "save_translations", "merge_glossary"]):
                            stage_emoji = "💾"
                        elif any(k in message for k in ["Готово", "completed", "finish"]):
                            stage_emoji = "✅"
                        elif "error" in message.lower() or "ошибка" in message.lower():
                            stage_emoji = "❌"
                        elif any(k in message for k in ["inpaint", "mask", "Background"]):
                            stage_emoji = "🎨"
                        if total and current:
                            await reporter.update(f"{stage_emoji} {manga_name} — гл. {ch_str}: {message} [{current}/{total}]")
                        else:
                            await reporter.update(f"{stage_emoji} {manga_name} — гл. {ch_str}: {message}")

                    pipeline.on_progress(_on_progress)
                    page_paths = await pipeline.process_chapter(
                        mangadex_manga_id=manga_id,
                        chapter_number=ch_str,
                        source_lang=source_lang,
                        target_lang="en",
                        source=source_name,
                    )
                    await pipeline.close()

                    if not page_paths:
                        await reporter.finish(f"❌ {title['name']} — гл. {ch_str}: не удалось перевести")
                        await bot.send_message(
                            chat_id,
                            f"❌ {title['name']} — гл. {ch_str}: не удалось перевести",
                        )
                        clear_reporter(chat_id, manga_id, ch_str)
                        continue

                    zip_path = Path("temp") / f"auto_{manga_id[:8]}_{ch_str}.zip"
                    zip_path.parent.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        for p in page_paths:
                            zf.write(p, arcname=Path(p).name)

                    await bot.send_document(
                        chat_id,
                        document=FSInputFile(str(zip_path)),
                        caption=f"{title['name']} — глава {ch_str}",
                    )

                    zip_path.unlink(missing_ok=True)

                    title["last_chapter"] = ch_str
                    title["chapters_count"] = max(
                        title.get("chapters_count", 0),
                        int(float(ch_str)),
                    )
                    save_config()

                    await bot.send_message(
                        chat_id,
                        f"✅ {title['name']} — гл. {ch_str}: отправлено",
                    )
                    await reporter.finish(f"✅ {title['name']} — гл. {ch_str}: отправлено")
                    clear_reporter(chat_id, manga_id, ch_str)

                except Exception as e:
                    logger.exception(f"auto: ошибка главы {ch_str}: {e}")
                    record_error()
                    try:
                        await reporter.finish(f"❌ {title['name']} — гл. {ch_str}: {e}")
                        clear_reporter(chat_id, manga_id, ch_str)
                    except Exception:
                        pass
                    await bot.send_message(
                        chat_id,
                        f"❌ {title['name']} — гл. {ch_str}: {e}",
                    )
                    continue

            await source.close()

        except Exception as e:
            logger.exception(f"auto: ошибка тайтла {title.get('name')}: {e}")
            record_error()


async def scheduler_loop(bot: Bot):
    await asyncio.sleep(30)
    while True:
        now = datetime.now(timezone.utc)
        next_hour = ((now.hour // 6) + 1) * 6
        next_run = now.replace(hour=next_hour % 24, minute=0, second=0, microsecond=0)
        if next_hour >= 24:
            next_run = next_run.replace(day=next_run.day + 1)

        sleep_seconds = (next_run - now).total_seconds() + 10
        logger.info(f"auto: след. проверка через {sleep_seconds / 3600:.1f}ч (в {next_run.isoformat()} UTC)")
        await asyncio.sleep(sleep_seconds)

        logger.info("auto: проверка новых глав...")
        try:
            await check_new_chapters(bot)
        except Exception as e:
            logger.exception(f"auto: ошибка: {e}")
            record_error()


async def queue_loop(bot: Bot):
    """Process the translation queue every 60 seconds and send results to Telegram."""
    chat_id = CONFIG.get("telegram", {}).get("chat_id")
    if not chat_id:
        logger.warning("queue_loop: chat_id не настроен, задачи будут выполняться без отправки в Telegram")
    await asyncio.sleep(5)
    while True:
        try:
            db = TranslationQueueDB()
            pending_tasks = db.get_pending_tasks()
            if pending_tasks:
                logger.info(f"Очередь: найдено {len(pending_tasks)} задач")
                for task_entry in pending_tasks:
                    manga_id = task_entry["manga_id"]
                    chapter_number = task_entry["chapter_number"]
                    source_lang = task_entry["source_lang"]
                    try:
                        logger.info(f"Очередь: перевожу {manga_id[:8]} гл. {chapter_number}")
                        db.update_task_status(manga_id, chapter_number, "processing")
                        reporter = get_reporter(chat_id, manga_id, chapter_number)
                        await reporter.start(f"📖 Гл. {chapter_number}: инициализация...")
                        pipeline = TranslationPipeline()

                        async def _on_progress(message: str, current: int, total: int):
                            stage_emoji = "🔄"
                            if any(k in message for k in ["Поиск", "find_chapter", "get_pages"]):
                                stage_emoji = "🌐"
                            elif any(k in message for k in ["page", "страница", "bubble", "bubble_clean", "Preprocess"]):
                                stage_emoji = "📄"
                            elif any(k in message for k in ["Перевод", "translate", "LLM"]):
                                stage_emoji = "🌐"
                            elif any(k in message for k in ["Сохранено", "save_translations", "merge_glossary"]):
                                stage_emoji = "💾"
                            elif any(k in message for k in ["Готово", "completed", "finish"]):
                                stage_emoji = "✅"
                            elif "error" in message.lower() or "ошибка" in message.lower():
                                stage_emoji = "❌"
                            elif any(k in message for k in ["inpaint", "mask", "Background"]):
                                stage_emoji = "🎨"
                            if total and current:
                                await reporter.update(f"{stage_emoji} Гл. {chapter_number}: {message} [{current}/{total}]")
                            else:
                                await reporter.update(f"{stage_emoji} Гл. {chapter_number}: {message}")

                        pipeline.on_progress(_on_progress)
                        page_paths = await pipeline.process_chapter(
                            mangadex_manga_id=manga_id,
                            chapter_number=chapter_number,
                            source_lang=source_lang,
                            target_lang="en",
                            source="mangakakalot",
                        )
                        await pipeline.close()
                        if page_paths:
                            db.update_task_status(manga_id, chapter_number, "completed")
                            logger.info(f"Очередь: гл. {chapter_number} переведена")
                            await reporter.finish(f"✅ Гл. {chapter_number}: переведена")
                            clear_reporter(chat_id, manga_id, chapter_number)
                            if chat_id and bot:
                                try:
                                    zip_path = Path("temp") / f"q_{manga_id[:8]}_ch_{chapter_number}.zip"
                                    zip_path.parent.mkdir(parents=True, exist_ok=True)
                                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                                        for p in page_paths:
                                            zf.write(p, arcname=Path(p).name)
                                    await bot.send_document(
                                        chat_id, document=FSInputFile(str(zip_path)),
                                        caption=f"Глава {chapter_number} — готова",
                                    )
                                    zip_path.unlink(missing_ok=True)
                                    await bot.send_message(chat_id, f"✅ Глава {chapter_number}: отправлено")
                                except Exception as e:
                                    logger.exception(f"Очередь: отправка гл. {chapter_number} не удалась: {e}")
                            else:
                                logger.info(f"Очередь: гл. {chapter_number} готова (chat_id не задан, отправка пропущена)")
                        else:
                            db.update_task_status(manga_id, chapter_number, "failed", "Нет страниц или не удалось перевести")
                            if chat_id and bot:
                                try:
                                    await bot.send_message(chat_id, f"❌ Глава {chapter_number}: не удалось перевести")
                                except Exception:
                                    pass
                    except Exception as e:
                        logger.exception(f"Очередь: ошибка гл. {chapter_number}: {e}")
                        record_error()
                        db.update_task_status(manga_id, chapter_number, "failed", str(e))
                        try:
                            await reporter.finish(f"❌ Гл. {chapter_number}: ошибка: {e}")
                            clear_reporter(chat_id, manga_id, chapter_number)
                        except Exception:
                            pass
                    await asyncio.sleep(10)  # delay between chapters
            db.close()
        except Exception as e:
            logger.exception(f"Очередь: ошибка цикла: {e}")
            record_error()
        await asyncio.sleep(60)


async def startup_translate(bot: Bot):
    chapter = os.environ.get("STARTUP_TRANSLATE_CHAPTER")
    if not chapter:
        return
    logger.info(f"startup: перевожу главу {chapter}...")
    titles = CONFIG.get("titles", [])
    chat_id = CONFIG.get("telegram", {}).get("chat_id")
    if not titles or not chat_id:
        logger.warning("startup: нет тайтлов или chat_id")
        return
    title = titles[0]
    try:
        pipeline = TranslationPipeline()
        page_paths = await pipeline.process_chapter(
            mangadex_manga_id=title.get("mangadex_id") or title.get("manga_id"),
            chapter_number=chapter,
            source_lang=title.get("source_lang", "ko"),
            target_lang="en",
            source=title.get("source", "mangakakalot"),
        )
        await pipeline.close()
        if not page_paths:
            await bot.send_message(chat_id, f"❌ Стартовый перевод гл. {chapter}: не удалось")
            return
        zip_path = Path("temp") / f"startup_{chapter}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in page_paths:
                zf.write(p, arcname=Path(p).name)
        await bot.send_document(
            chat_id, document=FSInputFile(str(zip_path)),
            caption=f"{title['name']} — глава {chapter}"
        )
        zip_path.unlink(missing_ok=True)
        title["last_chapter"] = chapter
        title["chapters_count"] = max(title.get("chapters_count", 0), int(float(chapter)))
        save_config()
        await bot.send_message(chat_id, f"✅ Глава {chapter}: готово!")
    except Exception as e:
        logger.exception(f"startup: ошибка: {e}")
        try:
            await bot.send_message(chat_id, f"❌ Стартовый перевод гл. {chapter}: {e}")
        except:
            pass


async def main():
    global bot
    bot = Bot(token=TG_BOT_TOKEN, session=build_session())
    from bot.utils.telegram_helpers import set_bot
    set_bot(bot)
    dp = Dispatcher()
    dp.message.middleware(CommandResetState())

    routers = get_routers()
    logger.info("Loaded %d routers", len(routers))
    for router in routers:
        dp.include_router(router)

    logger.info("Bot starting...")
    validate_config()

    health_runner = await start_health_server()

    editor_runner = None
    if os.environ.get("EDITOR_PORT"):
        try:
            from editor.server import start_editor_server
            editor_runner = await start_editor_server()
            logger.info(f"Editor server started on :{os.environ.get('EDITOR_PORT')}")
        except Exception as e:
            logger.warning(f"Editor server failed to start: {e}")

    dashboard_runner = None
    if os.environ.get("DASHBOARD_PORT"):
        try:
            from dashboard.server import start_dashboard_server
            dashboard_runner = await start_dashboard_server()
            logger.info(f"Dashboard server started on :{os.environ.get('DASHBOARD_PORT')}")
        except Exception as e:
            logger.warning(f"Dashboard server failed to start: {e}")

    mark_bot_started()
    asyncio.create_task(_keepalive())
    asyncio.create_task(scheduler_loop(bot))
    asyncio.create_task(queue_loop(bot))
    asyncio.create_task(startup_translate(bot))

    loop = asyncio.get_running_loop()

    async def _stop_polling():
        logger.info("Shutdown requested, stopping polling...")
        await dp.stop_polling()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(_stop_polling()))
        except NotImplementedError:
            pass

    try:
        await dp.start_polling(bot)
    except asyncio.CancelledError:
        logger.info("Polling cancelled, shutting down gracefully...")
    finally:
        await dp.stop_polling()
        await stop_health_server(health_runner)
        if editor_runner:
            try:
                await editor_runner.cleanup()
            except Exception:
                pass
        if dashboard_runner:
            try:
                await dashboard_runner.cleanup()
            except Exception:
                pass
        await bot.session.close()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
