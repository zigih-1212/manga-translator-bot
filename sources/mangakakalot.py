"""Mangakakalot source via MangaHook API — Solo Leveling 200 chapters."""
import asyncio
import logging
import httpx
from .base import BaseSource, MangaResult, Chapter, Page

log = logging.getLogger("manga_translator")

API_BASE = "https://mangahook-api.vercel.app/api"


class MangakakalotSource:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def _client_get(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def search(self, title: str) -> list[MangaResult]:
        """Search manga by title."""
        try:
            client = await self._client_get()
            r = await client.get(f"{API_BASE}/search", params={"query": title})
            if r.status_code != 200:
                return []
            data = r.json()
            results = []
            for item in data.get("results", []):
                results.append(MangaResult(
                    id=item.get("id", ""),
                    title=item.get("title", ""),
                    alt_titles=item.get("alt_titles", []),
                    description=item.get("description", ""),
                    status=item.get("status", ""),
                    year=item.get("year"),
                    cover_url=item.get("image", item.get("cover_url")),
                    source="mangakakalot",
                    original_language="ko",
                ))
            return results
        except Exception as e:
            log.warning("Mangakakalot search error: %s", e)
            return []

    async def get_chapters(self, manga_id: str, lang: str = "en") -> list[Chapter]:
        """Get chapter list."""
        try:
            client = await self._client_get()
            r = await client.get(f"{API_BASE}/manga/{manga_id}/chapters")
            if r.status_code != 200:
                return []
            data = r.json()
            chapters = []
            for item in data.get("chapters", []):
                chapters.append(Chapter(
                    id=item.get("id", ""),
                    number=item.get("number", ""),
                    title=item.get("title", ""),
                    volume=item.get("volume"),
                    pages_count=item.get("pages_count", 0),
                    translated_language=lang,
                ))
            return chapters
        except Exception as e:
            log.warning("Mangakakalot get_chapters error: %s", e)
            return []

    async def get_pages(self, chapter_id: str) -> list[Page]:
        """Get page images for chapter."""
        try:
            client = await self._client_get()
            r = await client.get(f"{API_BASE}/chapter/{chapter_id}")
            if r.status_code != 200:
                return []
            data = r.json()
            pages = []
            for i, url in enumerate(data.get("pages", data.get("images", []))):
                pages.append(Page(
                    url=url,
                    index=i,
                    width=0,
                    height=0,
                ))
            return pages
        except Exception as e:
            log.warning("Mangakakalot get_pages error: %s", e)
            return []

    async def download_page(self, page: Page) -> bytes:
        """Download page image."""
        client = await self._client_get()
        r = await client.get(page.url)
        return r.content

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
