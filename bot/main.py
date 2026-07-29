import asyncio
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
import httpx
import zipfile
from aiohttp import ClientSession
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import FSInputFile
from config import TG_BOT_TOKEN, TG_PROXY_URL, COLAB_URL, CONFIG, save_config
from bot.handlers import start_router, titles_router, translate_router, status_router
from sources.mangadex import MangaDexSource
from translator.pipeline import TranslationPipeline

HTTP_PROXY = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL = 600


async def _keepalive():
    if not COLAB_URL:
        return
    proxy = HTTP_PROXY or None
    while True:
        try:
            async with httpx.AsyncClient(proxy=proxy, timeout=10) as c:
                r = await c.get(f"{COLAB_URL}/health")
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

    for title in titles:
        manga_id = title.get("mangadex_id")
        source_lang = title.get("source_lang", "ko")
        last_chapter = title.get("last_chapter", "0")

        try:
            source = MangaDexSource()
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
                await bot.send_message(
                    chat_id,
                    f"🔄 {title['name']} — гл. {ch_str}: перевод...",
                )

                try:
                    pipeline = TranslationPipeline()
                    page_paths = await pipeline.process_chapter(
                        mangadex_manga_id=manga_id,
                        chapter_number=ch_str,
                        source_lang=source_lang,
                        target_lang="en",
                    )
                    await pipeline.close()

                    if not page_paths:
                        await bot.send_message(
                            chat_id,
                            f"❌ {title['name']} — гл. {ch_str}: не удалось перевести",
                        )
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

                except Exception as e:
                    logger.exception(f"auto: ошибка главы {ch_str}: {e}")
                    await bot.send_message(
                        chat_id,
                        f"❌ {title['name']} — гл. {ch_str}: {e}",
                    )
                    continue

            await source.close()

        except Exception as e:
            logger.exception(f"auto: ошибка тайтла {title.get('name')}: {e}")


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


async def main():
    bot = Bot(token=TG_BOT_TOKEN, session=build_session(), default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher()

    dp.include_router(start_router)
    dp.include_router(titles_router)
    dp.include_router(translate_router)
    dp.include_router(status_router)

    logger.info("Bot starting...")
    asyncio.create_task(_keepalive())
    asyncio.create_task(scheduler_loop(bot))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
