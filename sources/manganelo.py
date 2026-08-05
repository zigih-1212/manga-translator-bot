"""MangaNelo source — HTML scraping, Solo Leveling 200+ chapters."""
import asyncio
import logging
import re
import httpx
from .base import BaseSource, MangaResult, Chapter, Page

log = logging.getLogger("manga_translator")

BASE_URL = "https://manganelo.com"
SEARCH_URL = f"{BASE_URL}/search/story"


class MangaNeloSource:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _client_get(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
            )
        return self._client

    async def search(self, title: str) -> list[MangaResult]:
        """Search manga by title."""
        try:
            client = await self._client_get()
            r = await client.get(SEARCH_URL, params={"search": title})
            if r.status_code != 200:
                return []
            html = r.text
            results = []
            # Parse search results
            pattern = r'<div class="search-story-item">.*?<a[^>]*href="([^"]+)"[^>]*title="([^"]+)".*?<img[^>]*src="([^"]+)"'
            for match in re.finditer(pattern, html, re.DOTALL):
                url, name, img = match.groups()
                manga_id = url.rstrip("/").split("/")[-1]
                results.append(MangaResult(
                    id=manga_id,
                    title=name,
                    alt_titles=[],
                    description="",
                    status="",
                    year=None,
                    cover_url=img,
                    source="manganelo",
                    original_language="ko",
                ))
            return results
        except Exception as e:
            log.warning("MangaNelo search error: %s", e)
            return []

    async def get_chapters(self, manga_id: str, lang: str = "en") -> list[Chapter]:
        """Get chapter list."""
        try:
            client = await self._client_get()
            r = await client.get(f"{BASE_URL}/manga/{manga_id}")
            if r.status_code != 200:
                return []
            html = r.text
            chapters = []
            # Parse chapter list
            pattern = r'<a[^>]*class="chapter-name[^"]*"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
            for match in re.finditer(pattern, html):
                url, title = match.groups()
                chapter_id = url.rstrip("/").split("/")[-1]
                # Extract chapter number
                num_match = re.search(r'chapter[_-]?(\d+(?:\.\d+)?)', chapter_id, re.I)
                number = num_match.group(1) if num_match else chapter_id
                chapters.append(Chapter(
                    id=chapter_id,
                    number=number,
                    title=title.strip(),
                    volume=None,
                    pages_count=0,
                    translated_language=lang,
                ))
            # Reverse to get latest first
            chapters.reverse()
            return chapters
        except Exception as e:
            log.warning("MangaNelo get_chapters error: %s", e)
            return []

    async def get_pages(self, chapter_id: str) -> list[Page]:
        """Get page images for chapter."""
        try:
            client = await self._client_get()
            r = await client.get(f"{BASE_URL}/chapter/{chapter_id}")
            if r.status_code != 200:
                return []
            html = r.text
            pages = []
            # Parse page images
            pattern = r'<img[^>]*class="chapter-content[^"]*"[^>]*src="([^"]+)"'
            for i, match in enumerate(re.finditer(pattern, html)):
                url = match.group(1)
                pages.append(Page(
                    url=url.strip(),
                    index=i,
                    width=0,
                    height=0,
                ))
            return pages
        except Exception as e:
            log.warning("MangaNelo get_pages error: %s", e)
            return []

    async def download_page(self, page: Page) -> bytes:
        """Download page image."""
        client = await self._client_get()
        r = await client.get(page.url)
        return r.content

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
