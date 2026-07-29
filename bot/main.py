import asyncio
import logging
import os
import httpx
from aiohttp import ClientSession
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.client.session.base import BaseSession
from config import TG_BOT_TOKEN, TG_PROXY_URL, COLAB_URL
from bot.handlers import start_router, titles_router, translate_router, status_router

HTTP_PROXY = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL = 600  # 10 minutes


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


async def main():
    bot = Bot(token=TG_BOT_TOKEN, session=build_session(), default=DefaultBotProperties(parse_mode=None))
    dp = Dispatcher()

    dp.include_router(start_router)
    dp.include_router(titles_router)
    dp.include_router(translate_router)
    dp.include_router(status_router)

    logger.info("Bot starting...")
    asyncio.create_task(_keepalive())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
