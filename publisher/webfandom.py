import asyncio
from pathlib import Path
from playwright.async_api import async_playwright, Browser, BrowserContext, Page as PlaywrightPage
from cfg import WEBFANDOM_ACCESS_TOKEN, WEBFANDOM_REFRESH_TOKEN, CONFIG


class WebFandomPublisher:
    BASE_URL = CONFIG["webfandom"]["base_url"]

    def __init__(self):
        self.playwright = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: PlaywrightPage | None = None

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        await self.context.add_cookies([
            {"name": "access_token", "value": WEBFANDOM_ACCESS_TOKEN,
             "domain": "webfandom.ru", "path": "/"},
            {"name": "refresh_token", "value": WEBFANDOM_REFRESH_TOKEN,
             "domain": "webfandom.ru", "path": "/"},
        ])
        self.page = await self.context.new_page()

    async def navigate(self, url: str):
        if not self.page:
            raise RuntimeError("Publisher not started")
        await self.page.goto(url, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(1)

    async def upload_chapter(
        self,
        title_url: str,
        chapter_number: str,
        pages: list[bytes],
        title: str = "",
    ) -> str | None:
        if not self.page:
            raise RuntimeError("Publisher not started")

        await self.navigate(title_url)
        await asyncio.sleep(2)

        create_btn = self.page.locator('a:has-text("Добавить главу"), button:has-text("Добавить главу")')
        if await create_btn.count() > 0:
            await create_btn.first.click()
            await self.page.wait_for_load_state("networkidle")
            await asyncio.sleep(2)

        number_input = self.page.locator('input[name*="number"], input[name*="chapter"], input[placeholder*="номер"]')
        if await number_input.count() > 0:
            await number_input.first.fill(chapter_number)

        file_input = self.page.locator('input[type="file"]')
        if await file_input.count() > 0:
            temp_dir = Path("temp") / f"upload_{chapter_number}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            file_paths = []
            for i, page_bytes in enumerate(pages):
                p = temp_dir / f"{i:03d}.png"
                p.write_bytes(page_bytes)
                file_paths.append(str(p))
            await file_input.first.set_input_files(file_paths)
            await asyncio.sleep(3)

        publish_btn = self.page.locator('button:has-text("Опубликовать"), button:has-text("Сохранить")')
        if await publish_btn.count() > 0:
            await publish_btn.first.click()
            await self.page.wait_for_load_state("networkidle")
            await asyncio.sleep(3)

        return self.page.url

    async def get_page(self) -> PlaywrightPage:
        return self.page

    async def close(self):
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
